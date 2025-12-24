import torch
import os

# ================= CẤU HÌNH NGƯỜI DÙNG =================
# Thay đường dẫn file .pth bạn muốn trích xuất vào đây
INPUT_CHECKPOINT = "checkpoints/pipeline_ep91.pth" 

# Thư mục sẽ lưu file mới
OUTPUT_DIR = "extracted_weights"
# ========================================================

def extract_weights():
    if not os.path.exists(INPUT_CHECKPOINT):
        print(f"❌ Lỗi: Không tìm thấy file {INPUT_CHECKPOINT}")
        return

    print(f"🔄 Đang load file: {INPUT_CHECKPOINT} ...")
    
    # Load toàn bộ state_dict (dùng map_location='cpu' để chạy được mọi nơi)
    full_state_dict = torch.load(INPUT_CHECKPOINT, map_location='cpu')
    
    # Khởi tạo các dictionary chứa trọng số mới
    yolo_dict = {}
    projector_dict = {}
    logit_scale_tensor = None
    
    
    for key, value in full_state_dict.items():
        if key.startswith('yolo_model.'):
            new_key = key.replace('yolo_model.', '', 1)
            yolo_dict[new_key] = value
            
        elif key.startswith('projector.'):
            new_key = key.replace('projector.', '', 1)
            projector_dict[new_key] = value
            
        elif key == 'logit_scale':
            logit_scale_tensor = value
            
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if yolo_dict:
        yolo_path = os.path.join(OUTPUT_DIR, "yolo_extracted.pt")
        torch.save(yolo_dict, yolo_path)
        print(f"✅ Đã lưu trọng số YOLO tại: {yolo_path}")
        print(f"   -> Số lượng keys: {len(yolo_dict)}")
    else:
        print("⚠️ Không tìm thấy trọng số YOLO nào!")

    if projector_dict:
        adapter_checkpoint = {
            'projector': projector_dict,
            'logit_scale': logit_scale_tensor
        }
        adapter_path = os.path.join(OUTPUT_DIR, "adapter_extracted.pth")
        torch.save(adapter_checkpoint, adapter_path)
        print(f"✅ Đã lưu trọng số Adapter tại: {adapter_path}")
        print(f"   -> Projector keys: {len(projector_dict)}")
        if logit_scale_tensor is not None:
            print(f"   -> Logit Scale value: {logit_scale_tensor.item():.4f}")
    else:
        print("⚠️ Không tìm thấy trọng số Projector nào!")

if __name__ == "__main__":
    extract_weights()