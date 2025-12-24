import os
import glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class COCODataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # Lấy danh sách file ảnh
        self.image_paths = glob.glob(os.path.join(root_dir, "*.[jJ][pP][gG]")) + \
                           glob.glob(os.path.join(root_dir, "*.[pP][nN][gG]"))

        if len(self.image_paths) == 0:
            print(f"WARNING: NO IMAGE TO BE FOUND IN {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            # Chỉ mở ảnh và convert RGB, KHÔNG transform gì cả
            image = Image.open(img_path).convert('RGB')
            return image
        except Exception as e:
            print(f"ERROR READING FILE {img_path}: {e}")
            # Trả về ảnh đen nhỏ nếu lỗi để không crash
            return Image.new('RGB', (640, 640), (0, 0, 0))

def collate_fn(batch):
    """
    Hàm này nhận vào một list các kết quả từ __getitem__
    batch = [image1, image2, image3, ...] (List các PIL Image)
    Chúng ta giữ nguyên nó là List, không convert sang Tensor
    """
    return batch

def get_dataloader(root_dir, batch_size=16, num_workers=4, shuffle=True):
    dataset = COCODataset(root_dir=root_dir)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn, 
        pin_memory=False
    )
    return dataloader