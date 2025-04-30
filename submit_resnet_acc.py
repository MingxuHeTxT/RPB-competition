#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
submit_resnet_acc_with_cv.py

流程：
1. 定义 ResNet-18 两通道模型结构
2. 加载 best_resnet_aug_es_2.pth
3. 在 train.pkl 上做 3 折交叉验证并打印准确率
4. 用模型对 test.pkl 做预测
5. 将 0/1 → -1/+1，生成 submission_resnet_acc.csv
"""

import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from PIL import Image

# ─── 1) 定义模型结构（要和 train 脚本里完全一致） ─────────────────────────

class ResNet18TwoChannel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Linear(num_ftrs, 2)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

# ─── 2) Dataset 读取 (无增强，只 ToTensor+Normalize) ────────────────────────

class RPSTensorDataset(Dataset):
    def __init__(self, pkl_path, transform=None):
        data = pickle.load(open(pkl_path, 'rb'))
        self.imgs1 = np.stack(data['img1'])
        self.imgs2 = np.stack(data['img2'])
        labels = np.array(data['label'])  # +1 / -1
        self.labels = (labels == 1).astype(np.int64)  # 转 0/1
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

# ─── 3) 3 折交叉验证评估函数 ─────────────────────────────────────────────

def evaluate_cv(model, dataset, device, n_splits=3, batch_size=128):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    labels = dataset.labels
    fold_scores = []
    for fold, (_, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
        val_ds = Subset(dataset, val_idx)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        correct = total = 0
        model.eval()
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = correct / total
        print(f"🔵 Fold {fold} 验证 Acc: {acc:.4f}")
        fold_scores.append(acc)
    print(f"🔍 3 折 CV 平均 Acc: {np.mean(fold_scores):.4f}\n")
    return fold_scores

# ─── 4) 主流程 ───────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # (1) 准备模型并加载微调权重
    model = ResNet18TwoChannel().to(device)
    state = torch.load('best_resnet_aug_es_2.pth', map_location=device)
    model.load_state_dict(state)
    model.eval()

    # (2) 在 train.pkl 上做 3 折 CV
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_ds = RPSTensorDataset('train.pkl', transform=transform)
    print("开始 3 折交叉验证 …")
    evaluate_cv(model, train_ds, device, n_splits=3, batch_size=128)

    # (3) 对 test.pkl 做预测
    data = pickle.load(open('test.pkl', 'rb'))
    imgs1 = np.stack(data['img1'])
    imgs2 = np.stack(data['img2'])
    ids   = np.array(data['id'])

    bs = 128
    preds = []
    with torch.no_grad():
        for i in range(0, len(ids), bs):
            batch1 = imgs1[i:i+bs]
            batch2 = imgs2[i:i+bs]
            tx1 = torch.stack([transform(Image.fromarray(x.astype(np.uint8))) for x in batch1])
            tx2 = torch.stack([transform(Image.fromarray(x.astype(np.uint8))) for x in batch2])
            X = torch.cat([tx1, tx2], dim=1).to(device)  # (B,2,24,24)
            out = model(X)
            preds.append(out.argmax(dim=1).cpu().numpy())
    preds = np.concatenate(preds)

    # (4) 映射回 -1/+1 并保存 submission
    labels = np.where(preds == 1, 1, -1)
    df = pd.DataFrame({'id': ids, 'label': labels})
    df.to_csv('submission_resnet_acc.csv', index=False)
    print("✅ 已生成 submission_resnet_acc.csv (标签 -1/+1)")

if __name__ == '__main__':
    main()
