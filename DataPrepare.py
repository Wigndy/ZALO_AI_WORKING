from pathlib import Path
from typing import List, Dict
import json
import os

class DataPrepare:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.train_dir = self.root_dir / 'train'
        self.test_dir = self.root_dir / 'public_test'
        # Load annotation nếu cần (khi chạy train)
        self.annotations_file = self.train_dir / 'annotations' / 'annotations.json'
        if self.annotations_file.exists():
            with open(self.annotations_file, 'r') as f:
                self.annotations = json.load(f)
        else:
            self.annotations = []

    def get_annotations_for_sample(self, folder_name: str) -> Dict:
        for annotation in self.annotations:
            if annotation.get('video_id') == folder_name:
                return annotation
        return {}

    def get_class_folders(self, folder_dir) -> List[str]:
        class_folder_dir = folder_dir / 'samples'
        class_folders = []
        

        if class_folder_dir.exists():
            for item in class_folder_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    class_folders.append(item.name)

        return sorted(class_folders) 
    
    def get_support_image(self, folder_dir, class_folder) -> List[str]:
        object_images_dir = folder_dir / 'samples' / class_folder / 'object_images'
        image_files = []
        
        if object_images_dir.exists():
            for f in object_images_dir.iterdir():
                # Lọc file ảnh hợp lệ, bỏ qua folder con hoặc file ẩn
                if f.is_file() and not f.name.startswith('.'):
                    image_files.append(str(f))

        return sorted(image_files)
    
    def get_query_video(self, folder_dir, class_folder) -> str:
        video_path = folder_dir / 'samples' / class_folder / "drone_video.mp4"
        return str(video_path)
    
    def prepare_train_data(self) -> List[Dict]:
        # Logic tương tự train
        class_folders = self.get_class_folders(self.train_dir)
        train_data = []
        for folder in class_folders:
            support_images = self.get_support_image(self.train_dir, folder)
            query_video = self.get_query_video(self.train_dir, folder)
            sample = {
                'video_id': folder,
                'query_images': support_images,
                'num_query_images': len(support_images),
                'query_video': query_video
            }
            train_data.append(sample)
        return train_data
    
    def prepare_test_data(self) -> List[Dict]:
        # Logic tương tự test
        class_folders = self.get_class_folders(self.test_dir)
        test_data = []
        for folder in class_folders:
            support_images = self.get_support_image(self.test_dir, folder)
            query_video = self.get_query_video(self.test_dir, folder)
            sample = {
                'video_id': folder,
                'query_images': support_images,
                'num_query_images': len(support_images),
                'query_video': query_video
            }
            test_data.append(sample)
        return test_data