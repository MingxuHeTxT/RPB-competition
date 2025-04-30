#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_resnet_aug_es_acc.py

完整训练脚本（ResNet-18 + 数据增强，Early Stopping & LR 调度基于 Acc）：
1. 数据增强（随机旋转、水平翻转、随机裁剪、归一化）
2. ResNet-18 定义（输入通道改为 2，输出类别改为 2）
3. 训练 & 验证循环（打印每轮 loss/acc）
4. 学习率调度 (ReduceLROnPlateau 基于 val_acc 最大化)
5. Early Stopping（验证准确率无改善时提前终止）
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

# ─── 数据增强 & Dataset ────────────────────────────────────

augment = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(24, padding=2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class RPSAugDataset(Dataset):
    def __init__(self, pkl_path, transform=None):
        data = pickle.load(open(pkl_path, 'rb'))
        self.imgs1 = np.stack(data['img1'])
        self.imgs2 = np.stack(data['img2'])
        labels = np.array(data['label'])
        self.labels = torch.tensor((labels == 1).astype(np.int64))
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        im1 = Image.fromarray(self.imgs1[idx].astype(np.uint8))
        im2 = Image.fromarray(self.imgs2[idx].astype(np.uint8))
        if self.transform:
            im1 = self.transform(im1)
            im2 = self.transform(im2)
        x = torch.cat([im1, im2], dim=0)  # (2,24,24)
        y = self.labels[idx]
        return x, y

# ─── ResNet-18 定义 ─────────────────────────────────────────

class ResNet18TwoChannel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        backbone.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Linear(num_ftrs, 2)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

# ─── 训练 & 验证函数 ───────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return total_loss/total, correct/total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * X.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return total_loss/total, correct/total

# ─── 主流程 ───────────────────────────────────────────

def main():
    batch_size  = 64
    lr          = 1e-3
    weight_decay= 1e-4
    epochs      = 80
    patience_lr = 3   # LR 调度耐心（基于 val_acc）
    patience_es = 7   # Early Stopping 耐心（基于 val_acc）

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    full_ds = RPSAugDataset('train.pkl', transform=augment)
    n_train = int(len(full_ds)*0.8)
    train_ds, val_ds = random_split(full_ds, [n_train, len(full_ds)-n_train])
    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    va_loader = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model     = ResNet18TwoChannel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',          # 监控 val_acc，越大越好
        factor=0.5,
        patience=patience_lr
    )

    best_val_acc = 0.0
    no_improve   = 0

    for epoch in range(1, epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, tr_loader, criterion, optimizer, device)
        va_loss, va_acc = validate(model, va_loader, criterion, device)
        print(f"Epoch {epoch:2d} | "
              f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f} | "
              f"Val   Loss: {va_loss:.4f}, Acc: {va_acc:.4f}")

        # 基于 val_acc 调度学习率
        scheduler.step(va_acc)

        # Early Stopping & 保存最优模型（基于 Acc）
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            no_improve   = 0
            torch.save(model.state_dict(), 'best_resnet_aug_es_2.pth')
        else:
            no_improve += 1
            print(f"⏳ 验证准确率无改善 {no_improve}/{patience_es}")
            if no_improve >= patience_es:
                print("🚨 达到 Early Stopping 上限，终止训练")
                break

    print(f"\n🎉 最佳验证准确率: {best_val_acc:.4f}，模型保存在 best_resnet_aug_es.pth")

if __name__ == '__main__':
    main()
