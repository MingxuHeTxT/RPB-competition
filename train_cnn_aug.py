#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_cnn_aug.py

1. 加载并增强数据：随机旋转、水平翻转、填充裁剪 → ToTensor → 归一化
2. 定义 CNN 网络
3. 训练 & 验证循环
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# ─── 1. 数据增强 & Dataset ────────────────────────────────────

# 定义增强管道
augment = transforms.Compose([
    transforms.RandomRotation(10),          # ±10° 随机旋转
    transforms.RandomHorizontalFlip(),      # 随机水平翻转
    transforms.RandomCrop(24, padding=2),   # 随机裁剪到 24×24
    transforms.ToTensor(),                  # 转成 [0,1] 之间的张量 C×H×W
    transforms.Normalize((0.5,), (0.5,)),   # 归一化到 [-1,1]
])

class RPSAugDataset(Dataset):
    def __init__(self, pkl_path, transform=None):
        data = pickle.load(open(pkl_path, 'rb'))
        self.imgs1 = np.stack(data['img1'])  # (N,24,24)
        self.imgs2 = np.stack(data['img2'])
        labels = np.array(data['label'])     # (+1, -1)
        # 转成 0/1
        self.labels = torch.tensor((labels == 1).astype(np.int64))
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 取出 numpy 灰度图并转 PIL
        im1 = Image.fromarray(self.imgs1[idx].astype(np.uint8))
        im2 = Image.fromarray(self.imgs2[idx].astype(np.uint8))
        # 应用同样的增强
        if self.transform:
            im1 = self.transform(im1)   # Tensor 1×24×24
            im2 = self.transform(im2)
        # 拼成 2 通道
        x = torch.cat([im1, im2], dim=0)  # -> 2×24×24
        y = self.labels[idx]
        return x, y

# ─── 2. 简单 CNN ────────────────────────────────────────────

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2,2)
        self.fc1   = nn.Linear(64*6*6, 128)
        self.fc2   = nn.Linear(128, 2)
        self.drop  = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 24→12
        x = self.pool(F.relu(self.conv2(x)))  # 12→6
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)

# ─── 3. 训练 & 验证循环 ───────────────────────────────────

def train_one_epoch(model, loader, crit, opt, device):
    model.train()
    loss_sum, correct, total = 0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        opt.zero_grad()
        logits = model(X)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds==y).sum().item()
        total += y.size(0)
    return loss_sum/total, correct/total

def validate(model, loader, crit, device):
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = crit(logits, y)
            loss_sum += loss.item() * X.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds==y).sum().item()
            total += y.size(0)
    return loss_sum/total, correct/total

def main():
    # 超参
    batch_size = 64
    lr = 5e-4
    epochs = 15

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据集 & 划分
    full_ds = RPSAugDataset('train.pkl', transform=augment)
    train_n = int(len(full_ds)*0.8)
    val_n   = len(full_ds) - train_n
    train_ds, val_ds = random_split(full_ds, [train_n, val_n])
    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(val_ds, batch_size=batch_size)

    # 模型 & 优化器 & 损失
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练循环
    best_va_acc = 0.
    for ep in range(1, epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, tr_loader, criterion, optimizer, device)
        va_loss, va_acc = validate(model, va_loader, criterion, device)
        print(f"Epoch {ep:2d} | "
              f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f} | "
              f"Val   Loss: {va_loss:.4f}, Acc: {va_acc:.4f}")
        best_va_acc = max(best_va_acc, va_acc)

    print(f"\n🎉 数据增强后最佳验证 Acc: {best_va_acc:.4f}")

if __name__ == '__main__':
    main()
