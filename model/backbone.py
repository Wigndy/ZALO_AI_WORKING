import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from ultralytics import YOLO

class YOLO_CLIP_Pipeline(nn.Module):
    def __init__(self, yolo_version='yolov8m.pt', clip_version='ViT-B/32'):
        super().__init__()
        
        self.clip_model, self.clip_preprocess = clip.load(clip_version, device='cpu')
        
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        self.yolo = YOLO(yolo_version)
        self.yolo_model = self.yolo.model
        
        self.detect_layer = self.yolo_model.model[-1]
        self.captured_features = []
        self.detect_layer.register_forward_hook(self._hook_fn)
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 640, 640)
            self.yolo_model(dummy)
            p3, p4, p5 = self.captured_features
            total_channels = p3.shape[1] + p4.shape[1] + p5.shape[1]
            print(f"-> Detected Fusion Channels: {total_channels} (Auto-configured)")
            
        self.projector = nn.Sequential(
            nn.Conv2d(total_channels, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)
        
    def _hook_fn(self, module, inputs, outputs):
        self.captured_features = inputs[0]
        
    def fusion_layer(self, p3, p4, p5):
        target_h, target_w = p3.shape[2], p3.shape[3]
        p4_up = F.interpolate(p4, size=(target_h, target_w), mode='bilinear', align_corners=False)
        p5_up = F.interpolate(p5, size=(target_h, target_w), mode='bilinear', align_corners=False)
        fused = torch.cat([p3, p4_up, p5_up], dim=1) 
        out = self.projector(fused)
        return F.adaptive_avg_pool2d(out, (1, 1)).flatten(1)
    
    def forward(self, image_batch):
        clip_input = F.interpolate(image_batch, size=(224,224), model='bilinear', align_corners=False)
        
        with torch.no_grad():
            clip_feats = self.clip_model.encode_image(clip_input)
            clip_feats = clip_feats / clip_feats.norm(dim=1, keepdim=True)
        
        self.captured_features = []
        
        _ = self.yolo_model(image_batch)
        
        p3, p4, p5 = self.captured_features
        
        yolo_feats = self.fusion_layer(p3, p4, p5)
        yolo_feats = yolo_feats / yolo_feats.norm(dim=1, keepdim=True)
        
        return yolo_feats, clip_feats
        
        
def contrastive_loss(yolo_feats, clip_feats, logit_scale):
    logit_scale = logit_scale.exp()
    logits_per_image = logit_scale * yolo_feats @ clip_feats.t()
    logits_per_text = logits_per_image.t()
    
    batch_size = yolo_feats.shape[0]
    ground_truth = torch.arange(batch_size, dtype=torch.long, device=yolo_feats.device)
    
    loss_img - F.cross_entropy(logits_per_image, ground_truth)
    loss_txt = F.cross_entropy(logits_per_text, ground_truth)
    
    return (loss_img + loss_txt)/2