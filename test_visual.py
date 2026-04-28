"""
침수 영역 분할 추론 스크립트
- 학습된 best_model.pth 로 단일 이미지 추론
- 결과: 원본 / 마스크 / 오버레이 3종 저장
"""
 
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
 
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
 
# ──────────────────────────────────────────
# 설정 (여기만 수정)
# ──────────────────────────────────────────
class CFG:
    IMAGE_PATH  = "./test_images/test1.jpg"   # 테스트 이미지 경로
    MODEL_PATH  = "./checkpoints/best_model.pth"
    OUTPUT_DIR  = "./results"
    IMG_SIZE    = 512
    THRESHOLD   = 0.5    # 이 값 이상이면 '물' 로 판정
    ENCODER     = "resnet34"
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
 
 
# ──────────────────────────────────────────
# 전처리
# ──────────────────────────────────────────
def get_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
 
 
# ──────────────────────────────────────────
# 모델 로드
# ──────────────────────────────────────────
def load_model(model_path, encoder, device):
    model = smp.Unet(
        encoder_name    = encoder,
        encoder_weights = None,       # 추론 시엔 pretrained 불필요
        in_channels     = 3,
        classes         = 1,
        activation      = None,
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"모델 로드 완료: {model_path}")
    return model
 
 
# ──────────────────────────────────────────
# 추론
# ──────────────────────────────────────────
@torch.no_grad()
def predict(model, image_path, transform, device, threshold):
    # 이미지 로드
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없어요: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image_rgb.shape[:2]
 
    # 전처리
    aug   = transform(image=image_rgb)
    tensor = aug["image"].unsqueeze(0).to(device)   # (1,3,H,W)
 
    # 추론
    logit = model(tensor)                            # (1,1,H,W)
    prob  = torch.sigmoid(logit).squeeze().cpu().numpy()  # (H,W) 0~1
 
    # 원본 크기로 복원
    prob_full = cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    mask      = (prob_full > threshold).astype(np.uint8)  # 0 or 1
 
    water_ratio = mask.mean() * 100
    print(f"물 영역 비율: {water_ratio:.1f}%")
 
    return image_rgb, prob_full, mask
 
 
# ──────────────────────────────────────────
# 결과 시각화 & 저장
# ──────────────────────────────────────────
def save_results(image_rgb, prob, mask, output_dir, image_path):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
 
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"침수 영역 분석 결과 — {Path(image_path).name}", fontsize=14)
 
    # ① 원본
    axes[0].imshow(image_rgb)
    axes[0].set_title("원본 이미지", fontsize=12)
    axes[0].axis("off")
 
    # ② 확률 히트맵
    im = axes[1].imshow(prob, cmap="RdYlBu_r", vmin=0, vmax=1)
    axes[1].set_title("물 확률 히트맵", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
 
    # ③ 오버레이
    overlay = image_rgb.copy()
    water_pixels = mask.astype(bool)
    overlay[water_pixels]  = (overlay[water_pixels] * 0.4 +
                               np.array([30, 120, 220]) * 0.6).astype(np.uint8)
    axes[2].imshow(overlay)
    water_patch = mpatches.Patch(color=(30/255, 120/255, 220/255), label="물 영역")
    axes[2].legend(handles=[water_patch], loc="lower right", fontsize=10)
    axes[2].set_title(f"오버레이 (물 {mask.mean()*100:.1f}%)", fontsize=12)
    axes[2].axis("off")
 
    plt.tight_layout()
    out_path = Path(output_dir) / f"{stem}_result.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"결과 저장: {out_path}")
 
    # 마스크 단독 저장
    mask_path = Path(output_dir) / f"{stem}_mask.png"
    cv2.imwrite(str(mask_path), mask * 255)
    print(f"마스크 저장: {mask_path}")
 
    return str(out_path)
 
 
# ──────────────────────────────────────────
# 메인
# ──────────────────────────────────────────
def main():
    print(f"디바이스: {CFG.DEVICE}")
 
    model     = load_model(CFG.MODEL_PATH, CFG.ENCODER, CFG.DEVICE)
    transform = get_transform(CFG.IMG_SIZE)
 
    image_rgb, prob, mask = predict(
        model, CFG.IMAGE_PATH, transform, CFG.DEVICE, CFG.THRESHOLD
    )
 
    save_results(image_rgb, prob, mask, CFG.OUTPUT_DIR, CFG.IMAGE_PATH)
    print("완료! results/ 폴더를 확인하세요.")
 
 
if __name__ == "__main__":
    main()