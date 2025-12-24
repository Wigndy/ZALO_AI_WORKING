from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

class COCODataLoader(Dataset):
    def __init__(self, img_root, imgsz=640, max_images=None):
        self.img_paths = sorted(list(Path(img_root).glob("*.jpg")))
        if max_images is not None:
            self.img_paths = self.img_paths[:max_images]

       # * YOLO preproces
        self.yolo_transform = transforms.Compose([
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),   
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert("RGB") 

        yolo_img = self.yolo_transform(img)   

        return img, yolo_img, str(path) # * CLIP input, YOLO input, path
    
def coco_align_collate(batch):
    pil_imgs  = [b[0] for b in batch]
    yolo_imgs = torch.stack([b[1] for b in batch], dim=0)  # [B, 3, H, W]
    paths     = [b[2] for b in batch]
    return pil_imgs, yolo_imgs, paths