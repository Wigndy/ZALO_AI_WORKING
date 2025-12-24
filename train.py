import torch
import os
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from models.pipeline import YOLO_CLIP_Pipeline
from utils.loss import contrastive_loss, AdvancedAlignLoss
from utils.dataloader import get_dataloader
import copy

BATCH_SIZE = 160
LEARNING_RATE = 1.2e-4
EPOCHS = 260
IMAGE_SIZE = 640
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATASET_PATH = "./datasets/coco/train2017"
# NUM_WORKERS = os.cpu_count() - 1
NUM_WORKERS = 32

RESUME_PATH = "checkpoints/pipeline_full_ep250.pth"
def save_yolo_weights_ultralytics_style(pipeline_model, path):
    """
    Lưu trọng số YOLO theo định dạng chuẩn Ultralytics.
    File này có thể load bằng: model = YOLO('path.pt')
    """
    print(f"--> Saving Ultralytics-compatible weights to {path}")
    pipeline_model.eval()
    base_model = pipeline_model
    yolo_clone = copy.deepcopy(base_model.yolo_model).to('cpu')

    detection_layer = yolo_clone.model[-1]
    detection_layer._forward_hooks.clear()

    ckpt =  {
        'epoch': -1,
        'best_fitness':None,
        'model': yolo_clone.half(),
        'ema':None,
        'updates':None,
        'optimizer':None,
        'train_args':None,
        'date':None,
        'version':'8.0.0'
    }
    torch.save(ckpt, path)
    print("--> Save complete.")
    pipeline_model.train()


def main():
    print(f"RUNNING on DEVICE: {DEVICE}")
    if not os.path.exists(DATASET_PATH):
        print("Lỗi path data.")
        return
        
    train_loader = get_dataloader(DATASET_PATH, BATCH_SIZE, IMAGE_SIZE, NUM_WORKERS)
    model = YOLO_CLIP_Pipeline().to(DEVICE)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    # criterion = AdvancedAlignLoss(use_focal=True, gamma=2.0).cuda()
    criterion = HybridFusionAlignLoss(w_global=0.2, w_soft=0.3, w_hard=0.5, use_focal=True, gamma=2.0).cuda()

    if RESUME_PATH and os.path.exists(RESUME_PATH):
        print(f"--> Đang load checkpoint để train tiếp: {RESUME_PATH}")
        checkpoint = torch.load(RESUME_PATH, map_location=DEVICE)
        
        if 'model_state_dict' in checkpoint:
            # Đây là chuẩn checkpoint đầy đủ
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1 # Resume từ epoch tiếp theo
            print(f"--> Resume (Full) thành công! Bắt đầu từ Epoch {start_epoch + 1}")
        else:
            # Đây là trường hợp file cũ (chỉ có weights)
            model.load_state_dict(checkpoint)
            print(f"--> Resume (Weights Only) thành công! Lưu ý: Optimizer reset mới. Bắt đầu từ Epoch 1.")
            start_epoch = 0 # Nếu chỉ load weight, ta không biết epoch bao nhiêu, nên cẩn thận hoặc tự set tay
            # Nếu bạn biết file đó là ep91, hãy uncomment dòng dưới:
            
    else:
        print("--> Train từ đầu (Scratch).")
    start_epoch = 91
    model.train()
    for epoch in range(start_epoch, EPOCHS):
        total_loss = 0
        current_lr = optimizer.param_groups[0]['lr']
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)
        for batch_idx, images in enumerate(train_loader):
            # images = images.to(DEVICE)
            yolo_feats, clip_feats = model(images)
            loss = criterion(yolo_feats, clip_feats, model.logit_scale)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            loss_val = loss.item() # <--- Định nghĩa loss_val ở đây
            total_loss += loss_val
            progress_bar.set_postfix({
                "Loss": f"{loss_val:.4f}", 
                "LR": f"{current_lr:.2e}"
            })

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")
        if (epoch + 1) % 5 == 0 or (epoch + 1) == EPOCHS:
            os.makedirs("checkpoints", exist_ok=True)
            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }
            
            torch.save(checkpoint_dict, f"checkpoints/pipeline_full_latest.pth")
            torch.save(checkpoint_dict, f"checkpoints/pipeline_full_ep{epoch+1}.pth")
            
            torch.save(model.state_dict(), f"checkpoints/pipeline_weights_ep{epoch+1}.pth")
            save_yolo_weights_ultralytics_style(model, f"checkpoints/yolo_updated_ep{epoch+1}.pt")
        
if __name__ == "__main__":
    main()