#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cnn_svm_resnet_pipeline.py

流程：
1. 加载训练好的 ResNet-18 特征提取器 (best_resnet_aug_es.pth)
2. 用 ResNet 提取 train.pkl 和 test.pkl 特征 (512 维)
3. 在 train 特征上做 3 折 CV 验证 SVM (C=0.5, gamma='scale')
4. 在 train 特征上训练最终 SVM
5. 在 test 特征上预测，生成 submission_resnet_svm.csv
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import pandas as pd

# ─── Dataset 定义 ─────────────────────────────────────────

class RPSResNetDataset(Dataset):
    def __init__(self, pkl_path, transform=None, with_label=True):
        data = pickle.load(open(pkl_path, 'rb'))
        self.imgs1 = np.stack(data['img1'])   # (N,24,24)
        self.imgs2 = np.stack(data['img2'])
        self.with_label = with_label
        if with_label:
            self.labels = np.array(data['label'])
        else:
            self.ids = np.array(data['id'])
        self.transform = transform

    def __len__(self):
        return len(self.imgs1)

    def __getitem__(self, idx):
        im1 = Image.fromarray(self.imgs1[idx].astype(np.uint8))
        im2 = Image.fromarray(self.imgs2[idx].astype(np.uint8))
        if self.transform:
            im1 = self.transform(im1)
            im2 = self.transform(im2)
        x = torch.cat([im1, im2], dim=0)  # (2,24,24)
        if self.with_label:
            return x, self.labels[idx]
        else:
            return x, self.ids[idx]

# ─── 冻结 ResNet-18 特征提取器 ───────────────────────────

class FrozenResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # 修改第一层 conv 输入通道为 2
        resnet.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 去掉最后的 fc
        modules = list(resnet.children())[:-1]
        self.encoder = nn.Sequential(*modules)
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.encoder(x)           # (B,512,1,1)
        return x.view(x.size(0), -1)  # (B,512)

# ─── 特征提取函数 ───────────────────────────────────────

def extract_features(model, loader, device):
    model.eval()
    feats, ys = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            f = model(X)              # (B,512)
            feats.append(f.cpu().numpy())
            ys.append(y)
    return np.vstack(feats), np.hstack(ys)

# ─── 主流程 ───────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) 加载 ResNet-18 特征提取器
    net = FrozenResNet18().to(device)
    state = torch.load('best_resnet_aug_es_2.pth', map_location=device)
    net.load_state_dict(state, strict=False)

    # 2) 准备数据 & 提取 train 特征
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_ds     = RPSResNetDataset('train.pkl', transform=transform, with_label=True)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    X_train, y_train = extract_features(net, train_loader, device)
    print("训练特征形状：", X_train.shape)

    # 3) 3 折 CV 验证 SVM
    print("\n开始 3 折交叉验证 (C=0.5, gamma='scale')...")
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        svm_cv = SVC(kernel='rbf', C=0.5, gamma='scale')
        svm_cv.fit(X_train[tr_idx], y_train[tr_idx])
        acc = svm_cv.score(X_train[val_idx], y_train[val_idx])
        print(f"  Fold {fold} 验证准确率: {acc:.4f}")
        cv_scores.append(acc)
    print(f"平均 3 折 CV 准确率: {np.mean(cv_scores):.4f}")

    # 4) 在全量 train 特征上训练最终 SVM
    svm = SVC(kernel='rbf', C=0.5, gamma='scale')
    svm.fit(X_train, y_train)
    print("✅ 最终 SVM 训练完成")

    # 5) 提取 test 特征 & 预测
    test_ds     = RPSResNetDataset('test.pkl', transform=transform, with_label=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    net.eval()
    feats_test, ids = [], []
    with torch.no_grad():
        for X, id_batch in test_loader:
            X = X.to(device)
            f = net(X)
            feats_test.append(f.cpu().numpy())
            ids.append(id_batch.numpy())

    X_test = np.vstack(feats_test)
    ids    = np.hstack(ids)
    y_pred = svm.predict(X_test)

    # 6) 生成 submission
    sub = pd.DataFrame({'id': ids, 'label': y_pred})
    sub.to_csv('submission_resnet_svm.csv', index=False)
    print("✅ 已生成 submission_resnet_svm.csv")

if __name__ == '__main__':
    main()
