# utils/loss.py
import torch
import torch.nn.functional as F

def contrastive_loss(yolo_feats, clip_feats, logit_scale):

    yolo_feats = yolo_feats.half() 
    clip_feats = clip_feats.half()
    logit_scale = logit_scale.half()
    
    logit_scale = logit_scale.exp()

    B, C, d = yolo_feats.shape
    
    logits_per_image = torch.einsum('bcd,nc->bdn', yolo_feats, clip_feats) * logit_scale
    
    ground_truth = torch.arange(B, dtype=torch.long, device=yolo_feats.device)
    ground_truth_flat = ground_truth.repeat_interleave(d)


    logits_img_flat = logits_per_image.reshape(B * d, B)
    loss_img = F.cross_entropy(logits_img_flat, ground_truth_flat)


    logits_txt = logits_per_image.permute(2, 1, 0) 
    logits_txt_flat = logits_txt.reshape(B * d, B)
    loss_txt = F.cross_entropy(logits_txt_flat, ground_truth_flat)

    
    return (loss_img + loss_txt) / 2



import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedAlignLoss(nn.Module):
    def __init__(self, use_focal=True, gamma=2.0, alpha=0.25):
        super().__init__()
        self.use_focal = use_focal
        self.gamma = gamma 
        self.alpha = alpha 
        
        self.max_temp = 100.0 
        self.min_temp = 0.01

    def get_ground_truth(self, device, num_logits):
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        return labels

    def focal_loss(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, reduction='none', label_smoothing=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

    def forward(self, yolo_feats, clip_feats, logit_scale):
        yolo_feats = F.normalize(yolo_feats, dim=1).to(dtype=torch.float32)
        clip_feats = F.normalize(clip_feats, dim=1).to(dtype=torch.float32)
    
        logit_scale = torch.clamp(logit_scale.exp(), min=self.min_temp, max=self.max_temp)

        B, C, d = yolo_feats.shape
        sim_map = torch.einsum('bcd,nc->bdn', yolo_feats, clip_feats) * logit_scale
        
        logits_img_to_text, _ = sim_map.max(dim=1) 
        
        logits_text_to_img = logits_img_to_text.t()

        labels = self.get_ground_truth(yolo_feats.device, B)

        if self.use_focal:
            loss_i2t = self.focal_loss(logits_img_to_text, labels)
            loss_t2i = self.focal_loss(logits_text_to_img, labels)
        else:
            loss_i2t = F.cross_entropy(logits_img_to_text, labels, label_smoothing=self.alpha)
            loss_t2i = F.cross_entropy(logits_text_to_img, labels, label_smoothing=self.alpha)

        return (loss_i2t + loss_t2i) / 2



class HybridFusionAlignLoss(nn.Module):
    def __init__(self, 
                 w_global=0.2,    # Trọng số cho Global (Context)
                 w_soft=0.3,      # Trọng số cho Soft Local (Gradient Flow)
                 w_hard=0.5,      # Trọng số cho Hard Local (Discriminative Power - Quan trọng nhất)
                 use_focal=True, 
                 gamma=2.0, 
                 alpha=0.25):
        super().__init__()
        
        # Weights
        self.w_global = w_global
        self.w_soft = w_soft
        self.w_hard = w_hard
        
        # Focal Params
        self.use_focal = use_focal
        self.gamma = gamma
        self.alpha = alpha
        
        # Scale Clamping
        self.max_temp = 100.0
        self.min_temp = 0.01

    def get_ground_truth(self, device, num_logits):
        return torch.arange(num_logits, device=device, dtype=torch.long)

    def focal_loss(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, reduction='none', label_smoothing=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

    def forward(self, yolo_feats, clip_feats, logit_scale):
        yolo_feats = F.normalize(yolo_feats, p=2, dim=1).float() # [B, C, N]
        clip_feats = F.normalize(clip_feats, p=2, dim=1).float() # [B, C]
        
        logit_scale = torch.clamp(logit_scale.exp(), min=self.min_temp, max=self.max_temp)
        
        B, C, N = yolo_feats.shape
        labels = self.get_ground_truth(yolo_feats.device, B)

        # =================================================================
        # NHÁNH 1: GLOBAL ALIGNMENT (Học ngữ cảnh chung)
        # =================================================================
        # Ép trung bình toàn bộ feature map của YOLO phải khớp với CLIP
        yolo_global = yolo_feats.mean(dim=2) 
        yolo_global = F.normalize(yolo_global, p=2, dim=1)
        
        sim_global = torch.matmul(yolo_global, clip_feats.T) * logit_scale
        
        if self.use_focal:
            loss_global = (self.focal_loss(sim_global, labels) + 
                           self.focal_loss(sim_global.T, labels)) / 2
        else:
            loss_global = (F.cross_entropy(sim_global, labels) + 
                           F.cross_entropy(sim_global.T, labels)) / 2

        sim_map = torch.einsum('bcn,kc->bnk', yolo_feats, clip_feats) * logit_scale
        
        # =================================================================
        # NHÁNH 2: SOFT-LOCAL ALIGNMENT (LogSumExp - Gradient Flow)
        # =================================================================
        # LogSumExp giúp gradient lan truyền qua nhiều điểm ảnh tiềm năng, không chỉ 1 điểm.
        # Giúp model không bị kẹt cục bộ quá sớm.
        sim_soft = torch.logsumexp(sim_map, dim=1)
        
        if self.use_focal:
            loss_soft = (self.focal_loss(sim_soft, labels) + 
                         self.focal_loss(sim_soft.T, labels)) / 2
        else:
            loss_soft = (F.cross_entropy(sim_soft, labels) + 
                         F.cross_entropy(sim_soft.T, labels)) / 2

        # =================================================================
        # NHÁNH 3: HARD-LOCAL ALIGNMENT (Max + Focal - Discriminative)
        # =================================================================
        # Đây là phần lõi bạn muốn giữ lại: Chỉ quan tâm điểm tốt nhất.
        # Giúp tạo ra bounding box chính xác sau này.
        sim_hard, _ = sim_map.max(dim=1) # [B, B_clip]
        
        # Bắt buộc dùng Focal Loss ở đây để giải quyết hard negatives
        loss_hard = (self.focal_loss(sim_hard, labels) + 
                     self.focal_loss(sim_hard.T, labels)) / 2

        # =================================================================
        # TỔNG HỢP LOSS
        # =================================================================
        # Combine 3 thành phần lại
        total_loss = (self.w_global * loss_global) + \
                     (self.w_soft   * loss_soft) + \
                     (self.w_hard   * loss_hard)
                     
        return total_loss