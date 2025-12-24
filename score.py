import json
import os


def bbox_iou(b1, b2):
    xA = max(b1["x1"], b2["x1"])
    yA = max(b1["y1"], b2["y1"])
    xB = min(b1["x2"], b2["x2"])
    yB = min(b1["y2"], b2["y2"])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area <= 0:
        return 0.0

    area1 = (b1["x2"] - b1["x1"]) * (b1["y2"] - b1["y1"])
    # SỬA LỖI: area2 tính theo chiều rộng * chiều cao (x2-x1)*(y2-y1)
    area2 = (b2["x2"] - b2["x1"]) * (b2["y2"] - b2["y1"]) 

    union = area1 + area2 - inter_area
    
    if union <= 0:
        return 0.0

    return inter_area / union

def compute_st_iou(gt_bboxes_list, pred_bboxes_list):
    gt_by_frame = {b["frame"]: b for b in gt_bboxes_list}
    pr_by_frame = {b["frame"]: b for b in pred_bboxes_list}

    gt_frames = set(gt_by_frame.keys())
    pr_frames = set(pr_by_frame.keys())

    intersection = gt_frames & pr_frames
    union_frames = gt_frames | pr_frames

    if not union_frames:
        return 0.0

    sum_iou = 0.0
    for f in intersection:
        iou = bbox_iou(gt_by_frame[f], pr_by_frame[f])
        sum_iou += iou
    den = float(len(union_frames))

    return sum_iou / den

def load_data_map(json_path, data_type="gt"):
    """
    Đọc file json và trả về dictionary dạng:
    {
        "video_id_1": [list_of_bboxes],
        "video_id_2": [list_of_bboxes],
        ...
    }
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    data_map = {}
    
    # Xử lý format khác nhau giữa GT và Prediction nếu cần, 
    # nhưng theo đề bài thì cả 2 đều là list of objects.
    
    # Input data là một list các video objects
    for record in data:
        vid_id = record.get("video_id")
        
        # Lấy list bounding boxes
        # Ground Truth dùng key "annotations", Submission dùng key "detections"
        # Cả 2 đều chứa 1 list, lấy phần tử đầu tiên (object target)
        
        bboxes = []
        if data_type == "gt":
            if "annotations" in record and len(record["annotations"]) > 0:
                bboxes = record["annotations"][0].get("bboxes", [])
        else: # prediction
            if "detections" in record and len(record["detections"]) > 0:
                bboxes = record["detections"][0].get("bboxes", [])
                
        data_map[vid_id] = bboxes
        
    return data_map

def evaluate(ground_truth_file, prediction_file):
    print(f"Loading GT from: {ground_truth_file}")
    gt_map = load_data_map(ground_truth_file, data_type="gt")
    
    print(f"Loading Pred from: {prediction_file}")
    pred_map = load_data_map(prediction_file, data_type="pred")
    
    st_iou_scores = []
    video_ids = list(gt_map.keys())
    
    print(f"\nEvaluating {len(video_ids)} videos...")
    print("-" * 40)
    print(f"{'Video ID':<20} | {'ST-IoU':<10}")
    print("-" * 40)

    for vid_id in video_ids:
        gt_boxes = gt_map[vid_id]
        pred_boxes = pred_map.get(vid_id, [])
        
        score = compute_st_iou(gt_boxes, pred_boxes)
        st_iou_scores.append(score)
        
        print(f"{vid_id:<20} | {score:.4f}")

    # Tính Final Score (Trung bình cộng)
    final_score = sum(st_iou_scores) / len(st_iou_scores) if st_iou_scores else 0.0
    
    print("-" * 40)
    print(f"FINAL SCORE (Mean ST-IoU): {final_score:.6f}")
    return final_score

if __name__ == "__main__":
    
    gt_path = "observing/train/annotations/annotations.json"      # File chứa Ground Truth
    pred_path = "train_prediction/train_prediction.json"     # File chứa Submission
    
    if os.path.exists(gt_path) and os.path.exists(pred_path):
        evaluate(gt_path, pred_path)
    else:
        print("Vui lòng kiểm tra lại đường dẫn file json.")