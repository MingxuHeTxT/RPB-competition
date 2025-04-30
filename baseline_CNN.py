#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_cnn.py

1. 加载 train.pkl，做 80/20 划分
2. 定义带 BatchNorm 的三层卷积网络
3. 训练 & 验证循环（打印每轮 loss/acc）
4. 在 test.pkl 上预测，生成 submission.csv（+1/-1 标签）
"""

import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# ─── 1. 数据集定义 ────────────────────────────────────────

class RPSDataset(Dataset):
    def __init__(self, pkl_path):
        data = pickle.load(open(pkl_path, 'rb'))
        imgs1 = np.stack(data['img1'])  # (N, 24,24)
        imgs2 = np.stack(data['img2'])
        X = np.stack([imgs1, imgs2], axis=1)  # (N,2,24,24)
        self.X = torch.tensor(X, dtype=torch.float32) / 255.0
        # +1→1, -1→0
        labels = np.array(data['label'])
        self.y = torch.tensor((labels == 1).astype(np.int64))
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ─── 2. 模型定义 ─────────────────────────────────────────

class ImprovedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool  = nn.MaxPool2d(2,2)
        self.drop  = nn.Dropout(0.5)
        self.fc1   = nn.Linear(128 * 3 * 3, 256)
        self.fc2   = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 24→12
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 12→6
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  #  6→3
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)

# ─── 3. 训练 & 验证函数 ───────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.
    correct = 0
    total = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
    return running_loss/total, correct/total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            running_loss += loss.item() * X_batch.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    return running_loss/total, correct/total

# ─── 4. 主流程 ───────────────────────────────────────────

def main():
    # 超参数
    batch_size = 64
    lr = 5e-4
    epochs = 20
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据集 & 划分
    full_ds = RPSDataset('train.pkl')
    train_size = int(len(full_ds)*0.8)
    val_size   = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    # 模型、损失、优化器
    model = ImprovedCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.
    for epoch in range(1, epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:2d} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_cnn.pth')

    print(f"\n🎉 最佳验证准确率: {best_val_acc:.4f}")

    # ─── 5. 预测 test.pkl 并生成 submission.csv ────────────
    # >>> 如果当前阶段还处于测试，你可以注释掉以下测试预测部分 <<<
    test_ds = pickle.load(open('test.pkl','rb'))
    imgs1 = np.stack(test_ds['img1']).reshape(-1,24,24)
    imgs2 = np.stack(test_ds['img2']).reshape(-1,24,24)
    Xt = np.stack([imgs1, imgs2], axis=1)
    Xt = torch.tensor(Xt, dtype=torch.float32) / 255.0
    Xt = Xt.to(device)
    model.load_state_dict(torch.load('best_cnn.pth'))
    model.eval()
    with torch.no_grad():
        logits = model(Xt)
        preds = logits.argmax(dim=1).cpu().numpy()
    # 1→+1, 0→-1
    labels = np.where(preds==1, 1, -1)
    sub = pd.DataFrame({'id': test_ds['id'], 'label': labels})
    sub.to_csv('submission.csv', index=False)
    print("✅ submission.csv 已生成")

if __name__ == '__main__':
    main()
