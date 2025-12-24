import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from ultralytics import YOLO
import numpy as np 
from ultralytics.data.augment import LetterBox
class YOLO_CLIP_Pipeline(nn.Module):
    def __init__(self, yolo_version='yolov8m.pt', clip_version='ViT-B/32'):
        super().__init__()
        
        self.clip_model, self.clip_preprocess = clip.load(clip_version, device='cuda')
        
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        yolo_wrapper = YOLO(yolo_version)
        self.yolo_model = yolo_wrapper.model


        self.yolo_letterbox = LetterBox(new_shape=(640, 640), auto=False, stride=32)

        
        self.detect_layer = self.yolo_model.model[-1]
        self.captured_features = []
        
        self.hook_handle = self.detect_layer.register_forward_hook(self._hook_fn)
        
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
        ).to('cuda')
        
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)
        
    def _hook_fn(self, module, inputs, outputs):
        self.captured_features = inputs[0]

    def remove_hook(self):
        if self.hook_handle:
            self.hook_handle.remove()
            
    def register_hook(self):
        self.hook_handle = self.detect_layer.register_forward_hook(self._hook_fn)
        
    def fusion_layer(self, p3, p4, p5):
        target_h, target_w = p3.shape[2], p3.shape[3]
        p4_up = F.interpolate(p4, size=(target_h, target_w), mode='bilinear', align_corners=False)
        p5_up = F.interpolate(p5, size=(target_h, target_w), mode='bilinear', align_corners=False)
        fused = torch.cat([p3, p4_up, p5_up], dim=1) 
        out = self.projector(fused)
        return out.flatten(2)


    def prepare_yolo_input(self, image_list):
        processed_tensors = []
        
        for img in image_list:
            im0 = np.array(img) 
            im = self.yolo_letterbox(image=im0)         
            im = im.transpose((2, 0, 1))
            im = im[::-1] 
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im).float()
            im /= 255.0 
            processed_tensors.append(im)
            
        return torch.stack(processed_tensors).to('cuda')
    
    def forward(self, image_input):
        if isinstance(image_input, list):
            clip_batch = torch.stack([self.clip_preprocess(img) for img in image_input]).to('cuda')
            yolo_batch = self.prepare_yolo_input(image_input)
        
        with torch.no_grad():
            # clip_input = self.clip_preprocess(image_batch).unsqueeze(0).to('cuda')
            clip_feats = self.clip_model.encode_image(clip_batch)
            clip_feats = F.normalize(clip_feats, p=2, dim=1)
            # clip_feats = clip_feats / clip_feats.norm(dim=1, keepdim=True)
        
        self.captured_features = []
        
        _ = self.yolo_model(yolo_batch)
        
        p3, p4, p5 = self.captured_features
        
        yolo_feats = self.fusion_layer(p3, p4, p5)
        
        return yolo_feats, clip_feats
        