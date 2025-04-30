#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_cnn_aug_es.py

完整训练脚本：
1. 数据增强（随机旋转、水平翻转、随机裁剪、归一化）
2. 简单 CNN 定义
3. 训练 & 验证循环
4. 学习率调度(ReduceLROnPlateau)
5. Early Stopping(验证损失无改善时提前终止)
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

# ─── 数据增强 & Dataset ────────────────────────────────────

augment = transforms.Compose([
    transforms.RandomRotation(10),         # ±10° 随机旋转
    transforms.RandomHorizontalFlip(),     # 随机水平翻转
    transforms.RandomCrop(24, padding=2),  # 随机裁剪回 24×24
    transforms.ToTensor(),                 # 转成 Tensor [0,1]
    transforms.Normalize((0.5,), (0.5,))   # 归一化到 [-1,1]
])

class RPSAugDataset(Dataset):
    def __init__(self, pkl_path, transform=None):
        data = pickle.load(open(pkl_path, 'rb'))
        self.imgs1 = np.stack(data['img1'])    # (N,24,24)
        self.imgs2 = np.stack(data['img2'])
        labels = np.array(data['label'])       # +1 / -1
        self.labels = torch.tensor((labels==1).astype(np.int64))
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        im1 = Image.fromarray(self.imgs1[idx].astype(np.uint8))
        im2 = Image.fromarray(self.imgs2[idx].astype(np.uint8))
        if self.transform:
            im1 = self.transform(im1)
            im2 = self.transform(im2)
        x = torch.cat([im1, im2], dim=0)  # 2×24×24
        y = self.labels[idx]
        return x, y

# ─── 简单 CNN ────────────────────────────────────────────

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64*6*6, 128)
        self.fc2   = nn.Linear(128, 2)
        self.drop  = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 24→12
        x = self.pool(F.relu(self.conv2(x)))  # 12→6
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)

# ─── 训练&验证函数 ───────────────────────────────────────

def train_one_epoch(model, loader, crit, opt, device):
    model.train()
    total_loss = correct = total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        opt.zero_grad()
        logits = model(X)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return total_loss/total, correct/total

def validate(model, loader, crit, device):
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = crit(logits, y)
            total_loss += loss.item() * X.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return total_loss/total, correct/total

# ─── 主流程 ───────────────────────────────────────────

def main():
    # 超参数
    batch_size = 64
    lr = 5e-4
    epochs = 30
    patience_lr = 3   # 调度器耐心
    patience_es = 5   # 早停耐心

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据集划分
    full_ds = RPSAugDataset('train.pkl', transform=augment)
    n_train = int(len(full_ds)*0.8)
    n_val   = len(full_ds) - n_train
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(val_ds,   batch_size=batch_size)

    # 模型、损失、优化器、调度器
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,
        patience=patience_lr
    )

    best_val_loss = float('inf')
    best_val_acc  = 0.0   
    no_improve    = 0

    for epoch in range(1, epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, tr_loader, criterion, optimizer, device)
        va_loss, va_acc = validate(model, va_loader, criterion, device)
        print(f"Epoch {epoch:2d} | "
              f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f} | "
              f"Val   Loss: {va_loss:.4f}, Acc: {va_acc:.4f}")

        # 学习率调度
        scheduler.step(va_loss)

        # Early Stopping 检查
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            no_improve = 0
            torch.save(model.state_dict(), 'best_cnn_aug_es.pth')
        else:
            no_improve += 1
            print(f"⏳ 验证损失无改善 {no_improve}/{patience_es}")
            if no_improve >= patience_es:
                print("🚨 达到 Early Stopping 上限，终止训练")
                break
        if va_acc > best_val_acc:
            best_val_acc = va_acc

    print(f"\n🎉 最佳验证损失: {best_val_loss:.4f}，模型保存在 best_cnn_aug_es.pth")
    print(f"🎉 最佳验证准确率: {best_val_acc:.4f}")
    
if __name__ == '__main__':
    main()

