#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_cnn_aug2.py

升级版训练脚本：
1. 增强的数据增强（+ColorJitter + RandomErasing）
2. 更深的CNN（3个Conv+BatchNorm块）
3. 训练/验证循环 with ReduceLROnPlateau + EarlyStopping
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
    transforms.RandomRotation(15),            # ±15°
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(24, padding=2),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.RandomErasing(p=0.1, scale=(0.02,0.2))
])

class RPSAug2Dataset(Dataset):
    def __init__(self, pkl_path, transform=None):
        data = pickle.load(open(pkl_path, 'rb'))
        self.imgs1 = np.stack(data['img1'])
        self.imgs2 = np.stack(data['img2'])
        labels = np.array(data['label'])
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

# ─── 深度 CNN ────────────────────────────────────────────

class DeepCNN(nn.Module):
    def __init__(self):
        super().__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2)
            )
        self.block1 = conv_block(2, 32)    # 24→12
        self.block2 = conv_block(32, 64)   # 12→6
        self.block3 = conv_block(64, 128)  # 6→3
        self.fc1    = nn.Linear(128*3*3, 256)
        self.drop   = nn.Dropout(0.5)
        self.fc2    = nn.Linear(256, 2)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)

# ─── 训练 & 验证函数 ───────────────────────────────────────

def train_one_epoch(model, loader, crit, opt, device):
    model.train()
    total_loss = correct = total = 0
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        opt.zero_grad()
        logits = model(X)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        correct += preds.eq(y).sum().item()
        total += y.size(0)
    return total_loss/total, correct/total

def validate(model, loader, crit, device):
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for X,y in loader:
            X,y = X.to(device), y.to(device)
            logits = model(X)
            loss = crit(logits, y)
            total_loss += loss.item() * X.size(0)
            preds = logits.argmax(dim=1)
            correct += preds.eq(y).sum().item()
            total += y.size(0)
    return total_loss/total, correct/total

# ─── 主流程 ────────────────────────────────────────────

def main():
    # 超参
    batch_size = 64
    lr = 5e-4
    epochs = 50
    patience_lr = 3
    patience_es = 5
    print(">> train_cnn_aug2.py 正在运行")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    full = RPSAug2Dataset('train.pkl', transform=augment)
    n_train = int(len(full)*0.8)
    train_ds, val_ds = random_split(full, [n_train, len(full)-n_train])
    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    va_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4, pin_memory=True)

    model = DeepCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience_lr)

    best_val_acc = 0.0
    no_imp = 0

    for epoch in range(1, epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, tr_loader, criterion, optimizer, device)
        va_loss, va_acc = validate(model, va_loader, criterion, device)
        print(f"Epoch {epoch:2d} | Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f} | Val Loss: {va_loss:.4f}, Acc: {va_acc:.4f}")

        scheduler.step(va_loss)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            no_imp = 0
            torch.save(model.state_dict(), 'best_cnn_aug2.pth')
        else:
            no_imp += 1
            if no_imp >= patience_es:
                print("🚨 Early stopping")
                break

    print(f"\n🎉 Best Val Acc: {best_val_acc:.4f} (model→ best_cnn_aug2.pth)")

if __name__ == '__main__':
    main()
