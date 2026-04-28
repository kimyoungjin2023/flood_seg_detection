import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==========================================
# 0. 설정 및 모델 로드
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) U-Net++ (수면 영역 분할: EfficientNet-b4 백본)
unet_model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b4",
    encoder_weights=None,
    in_channels=3,
    classes=1
)
unet_model.load_state_dict(torch.load("path/to/unet_weights.pth")) # 실제 가중치 경로
unet_model.to(DEVICE)
unet_model.eval()

transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# 2) YOLO11s-seg (기준 객체 탐지: boot, hood 등)
yolo_model = YOLO("path/to/yolo11s-seg.pt") # 실제 가중치 경로

# ==========================================
# 파이프라인 함수 정의 (원본 이미지 기반)
# ==========================================

def step1_segment_water(image):
    """
    [Background & 수면 영역 분할 - 원본 이미지 기반]
    """
    augmented = transform(image=image)
    input_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = unet_model(input_tensor)
        prob_mask = torch.sigmoid(output).squeeze().cpu().numpy()
    
    # 원본 해상도로 리사이즈 및 이진화
    water_mask = cv2.resize(prob_mask, (image.shape[1], image.shape[0]))
    water_mask = (water_mask > 0.5).astype(np.uint8) * 255
    
    return water_mask

def step2_detect_objects(image):
    """
    [기준 객체 탐지 - 원본 이미지 기반]
    """
    results = yolo_model(image, conf=0.5)[0]
    
    objects = []
    if results.masks is not None:
        for i, (mask, box, cls) in enumerate(zip(results.masks.data, results.boxes.xyxy, results.boxes.cls)):
            obj_mask = mask.cpu().numpy()
            obj_mask = cv2.resize(obj_mask, (image.shape[1], image.shape[0]))
            obj_mask = (obj_mask > 0.5).astype(np.uint8) * 255
            
            objects.append({
                'class_id': int(cls),
                'class_name': yolo_model.names[int(cls)],
                'bbox': box.cpu().numpy().astype(int),
                'mask': obj_mask
            })
    return objects

def step3_calculate_depth(water_mask, objects):
    """
    [탐지된 기준 객체 침수심 계산 - 원본 이미지 기반]
    ※ 주의: 버드아이뷰 보정이 없으므로, 픽셀 면적 비율이 실제 높이 비율과 원근감에 의해 다를 수 있습니다.
    """
    estimated_depths = []
    
    for obj in objects:
        obj_mask = obj['mask']
        cls_name = obj['class_name']
        
        # 1. 겹침(Intersection) 영역 계산
        overlap = cv2.bitwise_and(water_mask, obj_mask)
        overlap_area = np.count_nonzero(overlap)
        obj_area = np.count_nonzero(obj_mask)
        
        if obj_area == 0: continue
        
        # 2. 객체가 물에 잠긴 픽셀 비율
        submerged_ratio = overlap_area / obj_area
        
        # 3. 기준 객체별 예상 높이 (예시 데이터)
        max_height_cm = 80 if cls_name == 'hood' else 50
        
        # 침수심 계산 (원근 왜곡을 감안한 보정 계수 도입이 필요할 수 있음)
        depth = max_height_cm * submerged_ratio 
        estimated_depths.append(depth)
    
    final_depth = np.mean(estimated_depths) if estimated_depths else 0
    return final_depth

# ==========================================
# 메인 실행 및 시각화
# ==========================================
def main(image_path):
    # 1. 원본 이미지 로드
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # 2. 수면 분할 (원본 이미지 그대로 사용)
    water_mask = step1_segment_water(original_image)
    
    # 3. 객체 탐지 (원본 이미지 그대로 사용)
    objects = step2_detect_objects(original_image)
    
    # 4. 침수심 계산
    final_depth = step3_calculate_depth(water_mask, objects)
    
    # [시각화 과정]
    result_vis = original_image.copy()
    
    # 수면 오버레이 (파란색)
    water_overlay = np.zeros_like(result_vis, dtype=np.uint8)
    water_overlay[water_mask == 255] = [0, 0, 255] 
    cv2.addWeighted(water_overlay, 0.4, result_vis, 1 - 0.4, 0, result_vis)
    
    # 객체 오버레이 (초록색) 및 바운딩 박스
    for obj in objects:
        obj_overlay = np.zeros_like(result_vis, dtype=np.uint8)
        obj_overlay[obj['mask'] == 255] = [0, 255, 0]
        cv2.addWeighted(obj_overlay, 0.6, result_vis, 1 - 0.6, 0, result_vis)
        
        x1, y1, x2, y2 = obj['bbox']
        cv2.rectangle(result_vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(result_vis, obj['class_name'], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # 텍스트 출력
    h, w = result_vis.shape[:2]
    text = f"Depth: ~{final_depth:.1f}cm"
    cv2.putText(result_vis, text, (w - 350, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    
    # 결과 저장
    result_vis_bgr = cv2.cvtColor(result_vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite("output_result_no_bev.jpg", result_vis_bgr)
    print(f"✅ 분석 완료! (버드아이뷰 제외) 최종 예상 침수심: {final_depth:.1f}cm")

if __name__ == "__main__":
    test_image_path = "test_flood_image.jpg"
    main(test_image_path)