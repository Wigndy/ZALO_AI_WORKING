import torch 
import clip 
from ultralytics import YOLO
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image

class CLIPEncoder:
    def __init__(self, device):
        self.device = device
        self.encoder, self.preprocess = clip.load("ViT-B/32", device=device)

        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
    
    def extract_features(self, image_path):
        image = Image.open(image_path)
        image = self.preprocess(image).unsqueeze(0).to(self.device)

        image_feats = self.encoder.encode_image(image)

        return image_feats
    
    @torch.no_grad()
    def encode_batch(self, pil_images):
        batch = torch.stack([self.preprocess(img) for img in pil_images]).to(self.device)
        feats = self.encoder.encode_image(batch)
        feats = F.normalize(feats, p=2, dim=1)

        return feats
    
class YOLOv8mEncoder(nn.Module):
    def __init__(self, weight_path="yolov8m.pt", device="cuda", imgsz=640):
        super().__init__()
        self.yolo = YOLO(weight_path)
        self.device = device
        self.imgsz = imgsz

        self.detect_model = self.yolo.model.to(device)
        self.detect_layer = self.detect_model.model[-1]

        self.feature_maps = None

        def hook(module, inputs, outputs):
            self.feature_maps = inputs[0]

        self.hook_handle = self.detect_layer.register_forward_hook(hook)

        self.projector = nn.Sequential(
            nn.Conv2d(144*3, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.Tanh()
        ).to(device)

    def train(self, mode: bool = True):
        self.training = mode
        self.detect_model.train(mode)
        self.projector.train(mode)
        return self

    def eval(self):
        return self.train(False)
    
    def fusion_feature_map(self, P3, P4, P5):
        target_h, target_w = P3.shape[2], P3.shape[3]
        p4_up = F.interpolate(P4, size=(target_h, target_w), mode='bilinear', align_corners=False)
        p5_up = F.interpolate(P5, size=(target_h, target_w), mode='bilinear', align_corners=False)
    
        fused = torch.cat([P3, p4_up, p5_up], dim=1)  # [B, 144*3, H, W]
        fused = self.projector(fused)
        fused = F.normalize(fused, dim=1)
        return fused  # [B, 512, H, W]
    
    def extract_feature_map(self, frame):
        _ = self.yolo.predict(source=frame, imgsz=self.imgsz, device=self.device, verbose=False)

        if self.feature_maps is None: print("Error to extract feature maps")

        return self.feature_maps
    
    def forward(self, images):
        self.feature_maps = None
        _ = self.detect_model(images)

        P3, P4, P5 = self.feature_maps
        fused = self.fusion_feature_map(P3, P4, P5)

        return fused

