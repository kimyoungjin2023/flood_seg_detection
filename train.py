"""
지하공간 침수 영역 분할 학습 코드
- 모델  : U-Net (backbone: ResNet34, pretrained on ImageNet)
- 데이터: Image(.jpg) + Binary Mask(.png)  흰색=물, 검정=배경
- 라이브러리: segmentation-models-pytorch (smp)
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ──────────────────────────────────────────
# 0. 설정값 (여기만 수정하면 됨)
# ──────────────────────────────────────────
class CFG:
    # 경로 — 에러 로그 기준 실제 경로로 수정
    DATA_DIR   = "./datasets/archive"  # images/ 와 masks/ 가 이 안에 있음
    CSV_PATH   = "./datasets/archive/metadata.csv"
    SAVE_DIR   = "./checkpoints"

    # 학습
    EPOCHS     = 80
    BATCH_SIZE = 4   # efficientnet-b4는 메모리 더 사용 (OOM 나면 2로 줄이세요)
    LR         = 3e-4
    IMG_SIZE   = 512
    NUM_WORKERS= 0     # Windows 멀티프로세싱 오류 방지     # Windows에서 멀티프로세싱 오류 방지 — 0 고정
    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

    # 모델
    ENCODER    = "efficientnet-b4" # 정확도 높은 버전
    WEIGHTS    = "imagenet"        # pretrained
    BATCH_SIZE_EFFECTIVE = 16  # gradient accumulation 으로 유효 배치 키움

    # 학습 전략
    VAL_RATIO  = 0.15
    PATIENCE   = 15                # Early stopping


# ──────────────────────────────────────────
# 1. Dataset
# ──────────────────────────────────────────
class FloodDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df        = df.reset_index(drop=True)
        self.data_dir  = Path(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path  = self.data_dir / "images" / self.df.loc[idx, "Image"]
        mask_path = self.data_dir / "masks"  / self.df.loc[idx, "Mask"]

        # 이미지 로드 (BGR → RGB)
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(
                f"이미지 파일을 읽을 수 없어요: {img_path}\n"
                f"CFG.DATA_DIR 을 실제 경로로 바꿔주세요. 현재값: {CFG.DATA_DIR}"
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 마스크 로드 (흰=1, 검=0 바이너리)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(
                f"마스크 파일을 읽을 수 없어요: {mask_path}\n"
                f"CFG.DATA_DIR 을 실제 경로로 바꿔주세요. 현재값: {CFG.DATA_DIR}"
            )
        # 이미지와 마스크 크기가 다를 경우 마스크를 이미지 크기에 맞게 리사이즈
        h, w = image.shape[:2]
        if mask.shape[0] != h or mask.shape[1] != w:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)   # 0.0 or 1.0

        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]

        return image, mask.unsqueeze(0)           # (C,H,W), (1,H,W)


# ──────────────────────────────────────────
# 2. Augmentation
# ──────────────────────────────────────────
def get_transforms(img_size, phase):
    if phase == "train":
        return A.Compose([
            A.Resize(img_size, img_size),
            # 기하학적 변환
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.4),
            A.RandomRotate90(p=0.4),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                               rotate_limit=30, p=0.5),
            A.ElasticTransform(p=0.2),
            A.GridDistortion(p=0.2),
            # 색상 변환 (침수 색상 다양성 대응)
            A.RandomBrightnessContrast(brightness_limit=0.3,
                                       contrast_limit=0.3, p=0.6),
            A.HueSaturationValue(hue_shift_limit=20,
                                 sat_shift_limit=40,
                                 val_shift_limit=20, p=0.5),
            A.RGBShift(p=0.3),
            A.CLAHE(p=0.3),
            # 노이즈/블러 (CCTV 화질 열화 시뮬레이션)
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.ISONoise(p=1.0),
                A.MultiplicativeNoise(p=1.0),
            ], p=0.3),
            A.OneOf([
                A.MotionBlur(p=1.0),
                A.GaussianBlur(p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.2),
            # 컷아웃 (일부 영역 가림 — 객체 가려져도 학습)
            A.CoarseDropout(max_holes=8, max_height=32,
                            max_width=32, p=0.3),
            A.Normalize(mean=(0.485,0.456,0.406),
                        std=(0.229,0.224,0.225)),
            ToTensorV2(),
        ])
    else:  # val / test
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485,0.456,0.406),
                        std=(0.229,0.224,0.225)),
            ToTensorV2(),
        ])


# ──────────────────────────────────────────
# 3. 모델
# ──────────────────────────────────────────
def build_model(encoder=CFG.ENCODER, weights=CFG.WEIGHTS):
    model = smp.UnetPlusPlus(   # U-Net++ : skip connection 강화 → 세밀한 경계 더 잘 잡음
        encoder_name        = encoder,
        encoder_weights     = weights,
        in_channels         = 3,
        classes             = 1,
        activation          = None,
    )
    return model


# ──────────────────────────────────────────
# 4. Loss & Metric
# ──────────────────────────────────────────
class CombinedLoss(nn.Module):
    """Dice + Focal Loss 조합
    - Dice  : 영역 전체 겹침 최대화
    - Focal : 어려운 픽셀(경계, 반사 영역)에 가중치 집중
    → 물 영역을 '거의 다' 잡으면서 경계도 선명하게
    """
    def __init__(self, dice_w=0.5, focal_w=0.5):
        super().__init__()
        self.dice_w  = dice_w
        self.focal_w = focal_w
        self.dice    = DiceLoss(mode="binary", from_logits=True)
        self.focal   = FocalLoss(mode="binary", gamma=2.0, alpha=0.75)

    def forward(self, pred, target):
        return self.dice_w  * self.dice(pred, target) + \
               self.focal_w * self.focal(pred, target)


def iou_score(pred_logits, target, threshold=0.5):
    pred  = (torch.sigmoid(pred_logits) > threshold).float()
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - inter
    iou   = ((inter + 1e-6) / (union + 1e-6)).mean()
    return iou.item()


# ──────────────────────────────────────────
# 5. Train / Validate 한 에폭
# ──────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, loss_fn, device, scheduler=None):
    model.train()
    total_loss, total_iou = 0.0, 0.0
    ACCUM = 4   # gradient accumulation steps (유효 배치 = BATCH_SIZE × ACCUM)

    optimizer.zero_grad()
    for step, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)

        preds = model(images)
        loss  = loss_fn(preds, masks) / ACCUM
        loss.backward()

        if (step + 1) % ACCUM == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * ACCUM
        total_iou  += iou_score(preds, masks)

    n = len(loader)
    return total_loss / n, total_iou / n


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss, total_iou = 0.0, 0.0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        preds = model(images)
        total_loss += loss_fn(preds, masks).item()
        total_iou  += iou_score(preds, masks)

    n = len(loader)
    return total_loss / n, total_iou / n


# ──────────────────────────────────────────
# 6. 학습 결과 시각화
# ──────────────────────────────────────────
def plot_history(history, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"],   label="val")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history["train_iou"], label="train")
    axes[1].plot(history["val_iou"],   label="val")
    axes[1].set_title("IoU")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(Path(save_dir) / "training_curve.png", dpi=120)
    plt.close()
    print(f"학습 곡선 저장: {save_dir}/training_curve.png")


# ──────────────────────────────────────────
# 7. 예측 시각화 (검증 샘플 3개)
# ──────────────────────────────────────────
@torch.no_grad()
def visualize_predictions(model, loader, device, save_dir, n=3):
    model.eval()
    images, masks = next(iter(loader))
    images, masks = images.to(device), masks.to(device)
    preds = torch.sigmoid(model(images)) > 0.5

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    for i in range(min(n, len(images))):
        img = (images[i].cpu() * std + mean).permute(1,2,0).clamp(0,1).numpy()
        axes[i][0].imshow(img);              axes[i][0].set_title("원본")
        axes[i][1].imshow(masks[i][0].cpu(), cmap="gray"); axes[i][1].set_title("GT 마스크")
        axes[i][2].imshow(preds[i][0].cpu(), cmap="gray"); axes[i][2].set_title("예측 마스크")
        for ax in axes[i]: ax.axis("off")

    plt.tight_layout()
    plt.savefig(Path(save_dir) / "predictions.png", dpi=120)
    plt.close()
    print(f"예측 시각화 저장: {save_dir}/predictions.png")


# ──────────────────────────────────────────
# 8. 메인
# ──────────────────────────────────────────
def main():
    os.makedirs(CFG.SAVE_DIR, exist_ok=True)
    print(f"디바이스: {CFG.DEVICE}")

    # --- 데이터 로드 & 분할 ---
    df = pd.read_csv(CFG.CSV_PATH)
    train_df, val_df = train_test_split(
        df, test_size=CFG.VAL_RATIO, random_state=42
    )
    print(f"Train: {len(train_df)}  |  Val: {len(val_df)}")

    # --- DataLoader ---
    train_ds = FloodDataset(train_df, CFG.DATA_DIR, get_transforms(CFG.IMG_SIZE, "train"))
    val_ds   = FloodDataset(val_df,   CFG.DATA_DIR, get_transforms(CFG.IMG_SIZE, "val"))

    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE,
                              shuffle=True,  num_workers=CFG.NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG.BATCH_SIZE,
                              shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)

    # --- 모델 / 옵티마이저 / 스케줄러 ---
    model   = build_model().to(CFG.DEVICE)
    loss_fn = CombinedLoss()

    # 인코더(pretrained)와 디코더 학습률 분리 — fine-tuning 핵심 기법
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": CFG.LR * 0.1},  # 인코더: 느리게
        {"params": model.decoder.parameters(), "lr": CFG.LR},         # 디코더: 빠르게
        {"params": model.segmentation_head.parameters(), "lr": CFG.LR},
    ], weight_decay=1e-4)

    # Warmup + CosineAnnealing
    from torch.optim.lr_scheduler import OneCycleLR
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[CFG.LR * 0.1, CFG.LR, CFG.LR],
        epochs=CFG.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,       # 전체의 10%를 warmup에 사용
        anneal_strategy="cos",
    )

    # --- 학습 루프 ---
    best_iou    = 0.0
    patience_cnt = 0
    history     = {"train_loss":[], "val_loss":[], "train_iou":[], "val_iou":[]}

    for epoch in range(1, CFG.EPOCHS + 1):
        tr_loss, tr_iou = train_one_epoch(model, train_loader, optimizer, loss_fn, CFG.DEVICE, scheduler)
        vl_loss, vl_iou = validate(model, val_loader, loss_fn, CFG.DEVICE)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_iou"].append(tr_iou)
        history["val_iou"].append(vl_iou)

        print(f"[{epoch:02d}/{CFG.EPOCHS}]  "
              f"Train Loss: {tr_loss:.4f}  IoU: {tr_iou:.4f}  |  "
              f"Val Loss: {vl_loss:.4f}  IoU: {vl_iou:.4f}")

        # 베스트 모델 저장
        if vl_iou > best_iou:
            best_iou = vl_iou
            patience_cnt = 0
            torch.save(model.state_dict(),
                       Path(CFG.SAVE_DIR) / "best_model.pth")
            print(f"  → 베스트 모델 저장 (IoU: {best_iou:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= CFG.PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # --- 결과 저장 ---
    plot_history(history, CFG.SAVE_DIR)
    visualize_predictions(model, val_loader, CFG.DEVICE, CFG.SAVE_DIR)
    print(f"\n학습 완료. 최고 Val IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()