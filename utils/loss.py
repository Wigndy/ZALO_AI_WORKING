import torch
import torch.nn as nn
import torch.nn.functional as F
        
def contrastive_loss(yolo_feats, clip_feats, logit_scale):
    logit_scale = logit_scale.exp()
    logits_per_image = logit_scale * yolo_feats @ clip_feats.t()
    logits_per_text = logits_per_image.t()
    
    batch_size = yolo_feats.shape[0]
    ground_truth = torch.arange(batch_size, dtype=torch.long, device=yolo_feats.device)
    
    loss_img - F.cross_entropy(logits_per_image, ground_truth)
    loss_txt = F.cross_entropy(logits_per_text, ground_truth)
    
    return (loss_img + loss_txt)/2