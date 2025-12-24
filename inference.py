from models.LTS import *
from DataPrepare import DataPrepare
from tqdm import tqdm
import cv2
import numpy as np
import torchvision.ops as ops
import json
import os
from score import *

def get_candidate_boxes(binary_heatmap, raw_heatmap, min_area=50, box_thresh=0.4):
    binary_mask = binary_heatmap.squeeze().detach().cpu().numpy().astype(np.uint8)
    raw_map_np = raw_heatmap.squeeze().detach().cpu().numpy()

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes, scores = [], []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if w * h < min_area: continue
        
        roi = raw_map_np[y:y+h, x:x+w]
        score = float(roi.max())
        if score < box_thresh: continue

        x1, y1 = x, y
        x2, y2 = x + w, y + h

        boxes.append([x1, y1, x2, y2])
        scores.append(score)

    if not boxes:
        return torch.empty((0, 4)), torch.empty((0,))

    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)

    return boxes, scores

def compute_heatmap(feature_map, query_embs):
    fm = feature_map.to(device=device, dtype=torch.float32)
    qe = query_embs.to(device=device, dtype=torch.float32)

    # * L2 norm to compute cosine simlarity
    feature_map_norm = F.normalize(fm, p=2, dim=1)
    query_embs_norm = F.normalize(qe, p=2, dim=1)

    # * Compute similarity between image point vector and query vector 
    similarity_map = torch.einsum('bchw,nc->bnhw', feature_map_norm, query_embs_norm)

    # * Max pooling theo query
    heatmap, _ = similarity_map.max(dim=1, keepdim=True)

    return heatmap


def process_video(video_id: str, video_path: str, frame_encoder, query_embs: torch.Tensor, threshold=0.35):
    cap = cv2.VideoCapture(video_path)

    # * video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    print(f"Video {video_id} properties:")
    print(f"fps = {fps}, frame size ({width},{height})")

    results = []
    frame_idx = 0

    # * Process bar 
    process_bar = tqdm(total=n_frames)
    global_max_score = -1.0
    # * Iterate video
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret: break

        # * Extract feature map from frame
        P3, P4, P5 = frame_encoder.extract_feature_map(frame)
        # P3, P4, P5 = P3.clone(), P4.clone(), P5.clone()
        feature_map = frame_encoder.fusion_feature_map(P3, P4, P5)

        # * Compute similarity heatmap
        heatmap = compute_heatmap(feature_map, query_embs) # (1, 1, H_fm, W_fm)

        # * Scale up to frame size
        frame_heatmap = F.interpolate(heatmap, size=(int(height), int(width)), mode="bilinear", align_corners=False)


        max_sim_val = frame_heatmap.max().item()
        if max_sim_val > global_max_score: global_max_score = max_sim_val

        if frame_idx < 5:
            print(f"Frame {frame_idx} Max Similarity: {max_sim_val:.4f}")
            
        if max_sim_val < threshold:
            frame_idx += 1
            process_bar.update(1)
            continue

        
        onehot_frame_heatmap = (frame_heatmap > threshold).float()

        # * Get bbox
        boxes, scores = get_candidate_boxes(onehot_frame_heatmap, frame_heatmap, box_thresh=threshold)
        
        # * Run NMS to filter overlaped bbox
        if boxes.numel() != 0:
            keep_idx = ops.nms(boxes, scores, 0.5)
            boxes = boxes[keep_idx]
            scores = scores[keep_idx]
            # * Get best box
            if boxes.shape[0]:
                best_idx = scores.argmax()
                x1, y1, x2, y2 = boxes[best_idx]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                results.append({
                    "frame": frame_idx, 
                    "x1": x1, 
                    "y1": y1, 
                    "x2": x2,
                    "y2": y2
                })

        frame_idx += 1
        process_bar.update(1)


    print(f"\n🔍 Video {video_id} Analysis: Global Max Score found = {global_max_score:.4f}")
    if global_max_score < threshold:
        print(f"⚠️ CẢNH BÁO: Max score ({global_max_score:.4f}) thấp hơn Threshold ({threshold}). Hãy giảm threshold!")
    process_bar.close()
    cap.release()

    return results


def load_custom_weights(encoder_instance, yolo_path, adapter_path):
    """
    Hàm load trọng số đã trích xuất vào class YOLOv8mEncoder mới
    """
    print(f"🔄 Bắt đầu load tham số cho {encoder_instance.__class__.__name__}...")
    device = encoder_instance.device

    # ================= 1. LOAD YOLO BACKBONE =================
    if os.path.exists(yolo_path):
        print(f"   -> Loading YOLO backbone từ: {yolo_path}")
        yolo_state_dict = torch.load(yolo_path, map_location=device)
        try:
            encoder_instance.detect_model.load_state_dict(yolo_state_dict, strict=True)
            print("   ✅ Load YOLO thành công!")
        except RuntimeError as e:
            print(f"   ❌ Lỗi shape YOLO: {e}")
            print("   ⚠️ Gợi ý: Kiểm tra xem version YOLO (n, s, m, l) có khớp nhau không.")
            return
    else:
        print(f"   ⚠️ Không tìm thấy file YOLO tại {yolo_path}")

    # ================= 2. LOAD PROJECTOR =================
    if os.path.exists(adapter_path):
        print(f"   -> Loading Adapter (Projector) từ: {adapter_path}")
        adapter_ckpt = torch.load(adapter_path, map_location=device)
        
        if 'projector' in adapter_ckpt:
            proj_weights = adapter_ckpt['projector']
            
            # Kiểm tra shape lớp đầu tiên để debug lỗi 144*3 vs 1536
            first_layer_weight = list(proj_weights.values())[0] # Key: '0.weight'
            trained_in_channels = first_layer_weight.shape[1]
            
            current_in_channels = encoder_instance.projector[0].in_channels
            
            if trained_in_channels != current_in_channels:
                print(f"   ❌ LỖI KÍCH THƯỚC PROJECTOR!")
                print(f"      - File save đang có input channels: {trained_in_channels}")
                print(f"      - Code hiện tại đang khai báo: {current_in_channels} (144*3?)")
                print(f"      -> HÃY SỬA LẠI class YOLOv8mEncoder dòng nn.Conv2d({trained_in_channels}, ...)")
                return

            encoder_instance.projector.load_state_dict(proj_weights)
            print("   ✅ Load Projector thành công!")
        else:
            print("   ⚠️ File adapter không chứa key 'projector'")
    else:
        print(f"   ⚠️ Không tìm thấy file Adapter tại {adapter_path}")

    print("🎉 Hoàn tất load model custom.")

def group_detections(raw_detections: list, max_gap: int = 1) -> list:
    if not raw_detections:
        return []

    # Sắp xếp lại theo frame để đảm bảo tính liên tục
    raw_detections.sort(key=lambda x: x["frame"])

    grouped_detections = []
    current_group = []

    for i, det in enumerate(raw_detections):
        if not current_group:
            current_group.append(det)
        else:
            previous_frame = current_group[-1]["frame"]
            current_frame = det["frame"]

            if current_frame - previous_frame <= max_gap:
                current_group.append(det)
            else:
                # Ngắt chuỗi, bắt đầu chuỗi mới
                grouped_detections.append({
                    "bboxes": current_group
                })
                current_group = [det]

    # Thêm nhóm cuối cùng
    if current_group:
        grouped_detections.append({
            "bboxes": current_group
        })
        
    return grouped_detections


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataloader = DataPrepare(root_dir="observing")
    test_set = dataloader.prepare_train_data()

    image_encoder = CLIPEncoder(device=device)
    frame_encoder = YOLOv8mEncoder(device=device)

    YOLO_CKPT = "extracted_weights/yolo_extracted.pt" 
    ADAPTER_CKPT = "extracted_weights/adapter_extracted.pth"

    load_custom_weights(frame_encoder, YOLO_CKPT, ADAPTER_CKPT)

    os.makedirs('./test_prediction', exist_ok=True)
    test_results = []

    for sample in test_set:
        video_id = sample["video_id"]
        sample_annotation = dataloader.get_annotations_for_sample(video_id)
        query_images = sample["query_images"]
        query_video = sample["query_video"]

        # * Query embedding
        query_embs = []
        for img_path in query_images:
            if os.path.isdir(img_path) or '.ipynb_checkpoints' in img_path:
                continue
            try:
                query_embs.append(image_encoder.extract_features(image_path=img_path))
            except Exception as e:
                print(f"⚠️ Lỗi khi đọc ảnh query {img_path}: {e}")
                continue
        query_embs = torch.cat(query_embs, dim=0)
        
        bboxes = process_video(video_id=video_id, video_path=query_video, frame_encoder=frame_encoder, query_embs=query_embs, threshold=0.14)


        grouped_detections = group_detections(bboxes, max_gap=1)
    
        sample_detection = {
            "video_id": video_id, 
            "detections": grouped_detections
        }

        with open(f'./train_prediction/{video_id}.json', 'w', encoding='utf-8') as f:
            json.dump(sample_detection, f, ensure_ascii=False, indent=4)

        test_results.append(sample_detection)

    with open('./train_prediction/train_prediction.json', 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=4)



