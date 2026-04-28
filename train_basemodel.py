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
from segmentation_models_pytorch.losses import DiceLoss

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
    EPOCHS     = 50
    BATCH_SIZE = 8
    LR         = 1e-4
    IMG_SIZE   = 512
    NUM_WORKERS= 0     # Windows 멀티프로세싱 오류 방지     # Windows에서 멀티프로세싱 오류 방지 — 0 고정
    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

    # 모델
    ENCODER    = "resnet34"        # 빠른 버전. 정확도↑원하면 resnet50/efficientnet-b4
    WEIGHTS    = "imagenet"        # pretrained

    # 학습 전략
    VAL_RATIO  = 0.15
    PATIENCE   = 10                # Early stopping


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
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(p=0.4),
            A.HueSaturationValue(p=0.3),
            A.GaussNoise(p=0.2),
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
    model = smp.Unet(
        encoder_name        = encoder,
        encoder_weights     = weights,
        in_channels         = 3,
        classes             = 1,            # 바이너리 (물/배경)
        activation          = None,         # sigmoid는 loss에서 처리
    )
    return model


# ──────────────────────────────────────────
# 4. Loss & Metric
# ──────────────────────────────────────────
class CombinedLoss(nn.Module):
    """Dice Loss + BCE Loss 조합 (침수 영역처럼 불균형 데이터에 효과적)"""
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha    = alpha
        self.dice     = DiceLoss(mode="binary", from_logits=True)
        self.bce      = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        return self.alpha * self.dice(pred, target) + \
               (1 - self.alpha) * self.bce(pred, target)


def iou_score(pred_logits, target, threshold=0.5):
    pred  = (torch.sigmoid(pred_logits) > threshold).float()
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - inter
    iou   = ((inter + 1e-6) / (union + 1e-6)).mean()
    return iou.item()


# ──────────────────────────────────────────
# 5. Train / Validate 한 에폭
# ──────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, total_iou = 0.0, 0.0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss  = loss_fn(preds, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
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
    model     = build_model().to(CFG.DEVICE)
    loss_fn   = CombinedLoss(alpha=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.EPOCHS, eta_min=1e-6
    )

    # --- 학습 루프 ---
    best_iou    = 0.0
    patience_cnt = 0
    history     = {"train_loss":[], "val_loss":[], "train_iou":[], "val_iou":[]}

    for epoch in range(1, CFG.EPOCHS + 1):
        tr_loss, tr_iou = train_one_epoch(model, train_loader, optimizer, loss_fn, CFG.DEVICE)
        vl_loss, vl_iou = validate(model, val_loader, loss_fn, CFG.DEVICE)
        scheduler.step()

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