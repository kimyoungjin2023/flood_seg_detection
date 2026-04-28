import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==========================================
# 0. 설정 및 모델 로드 (RTX 4060 Ti 최적화)
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) U-Net++ (수면 영역 분할) 로드
unet_model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b4",
    encoder_weights=None,
    in_channels=3,
    classes=1
)
unet_model.load_state_dict(torch.load("path/to/unet_weights.pth")) # 실제 가중치 경로 입력
unet_model.to(DEVICE)
unet_model.eval()

# U-Net++용 전처리 파이프라인
transform = A.Compose([
    A.Resize(512, 512), # 학습 시 사용한 resolution으로 변경
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# 2) YOLO11s-seg (기준 객체: boot, hood 등 탐지) 로드
yolo_model = YOLO("path/to/yolo11s-seg.pt") # 실제 가중치 경로 입력

# ==========================================
# 파이프라인 함수 정의
# ==========================================

def step1_create_bird_eye_view(image):
    """
    [카메라 보정 및 지면 평면 추정 -> 호모그래피 -> 버드아이뷰 생성]
    """
    h, w = image.shape[:2]
    
    # TODO: 실제 CCTV 설치 환경에 맞춘 4개의 소스 점과 목적지 점 입력 (캘리브레이션 데이터)
    src_pts = np.float32([[100, h], [w-100, h], [0, h//2], [w, h//2]]) 
    dst_pts = np.float32([[100, h], [w-100, h], [100, 0], [w-100, 0]])
    
    # 호모그래피 매트릭스 계산 및 투시 변환
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    bev_image = cv2.warpPerspective(image, matrix, (w, h))
    
    return bev_image, matrix

def step2_segment_water(bev_image):
    """
    [Background & 수면 영역 분할]
    """
    # 전처리 및 추론
    augmented = transform(image=bev_image)
    input_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = unet_model(input_tensor)
        prob_mask = torch.sigmoid(output).squeeze().cpu().numpy()
    
    # 원본 해상도로 리사이즈 및 이진화 (Threshold 0.5)
    water_mask = cv2.resize(prob_mask, (bev_image.shape[1], bev_image.shape[0]))
    water_mask = (water_mask > 0.5).astype(np.uint8) * 255
    
    return water_mask

def step3_detect_objects(bev_image):
    """
    [분할 공간에서 기준 객체 탐지 (YOLO11s-seg)]
    """
    # YOLO 추론 (conf, iou 등 파라미터 조절 가능)
    results = yolo_model(bev_image, conf=0.5)[0]
    
    objects = []
    if results.masks is not None:
        for i, (mask, box, cls) in enumerate(zip(results.masks.data, results.boxes.xyxy, results.boxes.cls)):
            # 마스크 리사이즈 및 포맷팅
            obj_mask = mask.cpu().numpy()
            obj_mask = cv2.resize(obj_mask, (bev_image.shape[1], bev_image.shape[0]))
            obj_mask = (obj_mask > 0.5).astype(np.uint8) * 255
            
            objects.append({
                'class_id': int(cls),
                'class_name': yolo_model.names[int(cls)],
                'bbox': box.cpu().numpy().astype(int),
                'mask': obj_mask
            })
    return objects

def step4_calculate_depth(water_mask, objects):
    """
    [탐지된 기준 객체 침수심 계산 및 가중치 통합]
    """
    estimated_depths = []
    
    for obj in objects:
        obj_mask = obj['mask']
        cls_name = obj['class_name']
        
        # 1. 수면 마스크와 객체 마스크의 겹침(Intersection) 영역 계산
        overlap = cv2.bitwise_and(water_mask, obj_mask)
        overlap_area = np.count_nonzero(overlap)
        obj_area = np.count_nonzero(obj_mask)
        
        if obj_area == 0: continue
        
        # 2. 객체가 물에 잠긴 비율
        submerged_ratio = overlap_area / obj_area
        
        # 3. 기하학적 침수심 산출 (TODO: 클래스별 실제 높이(cm) 데이터 매핑 필요)
        # 예시: hood(보닛) 높이가 지면에서 80cm라고 가정
        max_height_cm = 80 if cls_name == 'hood' else 50 # boot 등 다른 객체 조건 추가
        
        # 잠긴 비율 비례로 임시 계산 (실제 알고리즘에 맞춰 고도화 필요)
        depth = max_height_cm * submerged_ratio 
        estimated_depths.append(depth)
    
    # 4. 신뢰도 기반 통합 (단순 평균 예시, 필요시 가중 평균으로 변경)
    final_depth = np.mean(estimated_depths) if estimated_depths else 0
    return final_depth

# ==========================================
# 메인 실행 및 시각화 (출력)
# ==========================================
def main(image_path):
    # 이미지 로드
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # [파이프라인 실행]
    # 1. 버드아이뷰 생성
    bev_image, M = step1_create_bird_eye_view(original_image)
    
    # 2. 수면 분할
    water_mask = step2_segment_water(bev_image)
    
    # 3. 객체 탐지
    objects = step3_detect_objects(bev_image)
    
    # 4. 침수심 계산
    final_depth = step4_calculate_depth(water_mask, objects)
    
    # [시각화: Ex) 침수심: 약 20cm 출력]
    result_vis = bev_image.copy()
    
    # 수면 영역 파란색으로 오버레이
    water_overlay = np.zeros_like(result_vis, dtype=np.uint8)
    water_overlay[water_mask == 255] = [0, 0, 255] # Red in RGB -> Blue in BGR visualization if needed. Using Blue here [0,0,255]
    cv2.addWeighted(water_overlay, 0.4, result_vis, 1 - 0.4, 0, result_vis)
    
    # 객체 마스크 초록색으로 오버레이 및 바운딩 박스
    for obj in objects:
        obj_overlay = np.zeros_like(result_vis, dtype=np.uint8)
        obj_overlay[obj['mask'] == 255] = [0, 255, 0]
        cv2.addWeighted(obj_overlay, 0.6, result_vis, 1 - 0.6, 0, result_vis)
        
        x1, y1, x2, y2 = obj['bbox']
        cv2.rectangle(result_vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(result_vis, obj['class_name'], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # 최종 침수심 텍스트 출력 (우측 하단)
    h, w = result_vis.shape[:2]
    text = f"Depth: ~{final_depth:.1f}cm"
    cv2.putText(result_vis, text, (w - 300, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    
    # RGB to BGR for OpenCV display/save
    result_vis_bgr = cv2.cvtColor(result_vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite("output_result.jpg", result_vis_bgr)
    print(f"✅ 분석 완료! 최종 예상 침수심: {final_depth:.1f}cm (결과 이미지: output_result.jpg)")

if __name__ == "__main__":
    test_image_path = "test_flood_image.jpg" # 테스트할 이미지 경로
    main(test_image_path)