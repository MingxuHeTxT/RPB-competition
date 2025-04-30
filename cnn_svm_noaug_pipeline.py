#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cnn_svm_noaug_pipeline.py

流程：
1. 加载训练好的 CNN (SimpleCNN + best_cnn_noaug_es.pth)
2. 用 CNN 提取 train.pkl 和 test.pkl 特征 (128 维)
3. 在 train 特征上做 3 折交叉验证 SVM
4. 用最佳 SVM 模型在 test 特征上预测，生成 submission_noaug_svm.csv
"""

import pickle
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import pandas as pd

# ─── SimpleCNN 定义 (与 train_cnn_noaug_es.py 保持一致) ───────────────────

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

    def extract(self, x):
        # 提取 fc1-relu 之前的特征
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return F.relu(self.fc1(x))

# ─── Dataset for feature extraction ─────────────────────────────────────────

transform_feat = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class RPSFeatDataset(Dataset):
    def __init__(self, pkl_path, transform=None):
        data = pickle.load(open(pkl_path, 'rb'))
        self.imgs1 = np.stack(data['img1'])    # (N,24,24)
        self.imgs2 = np.stack(data['img2'])
        labels = data.get('label', None)
        self.ids    = data.get('id',    None)
        self.labels = np.array(labels) if labels is not None else None
        self.transform = transform

    def __len__(self):
        return len(self.imgs1)

    def __getitem__(self, idx):
        im1 = Image.fromarray(self.imgs1[idx].astype(np.uint8))
        im2 = Image.fromarray(self.imgs2[idx].astype(np.uint8))
        if self.transform is not None:
            im1 = self.transform(im1)
            im2 = self.transform(im2)
        x = torch.cat([im1, im2], dim=0)  # 2×24×24
        if self.labels is not None:
            return x, self.labels[idx]
        else:
            return x, self.ids[idx]

# ─── 提取特征函数 ────────────────────────────────────────────

def extract_features(model, loader, device):
    model.eval()
    feats = []
    ys    = []
    with torch.no_grad():
        for batch in loader:
            x, y = batch
            x = x.to(device)
            f = model.extract(x)           # (B,128)
            feats.append(f.cpu().numpy())
            ys.append(y.numpy())
    return np.vstack(feats), np.hstack(ys)

# ─── 主流程 ─────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 1. 加载 CNN
    cnn = SimpleCNN().to(device)
    cnn.load_state_dict(torch.load('best_cnn_noaug_es.pth', map_location=device))

    # 2. 提取训练集特征
    train_ds     = RPSFeatDataset('train.pkl', transform=transform_feat)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    X_train, y_train = extract_features(cnn, train_loader, device)
    print(f"训练特征形状： {X_train.shape}")

    # 3. 3 折交叉验证 SVM
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    val_scores = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        svm = SVC(kernel='rbf', C=0.85, gamma='scale')
        svm.fit(X_tr, y_tr)
        acc = svm.score(X_val, y_val)
        print(f"Fold {fold} 验证准确率: {acc:.4f}")
        val_scores.append(acc)
    print(f"平均验证准确率: {np.mean(val_scores):.4f}")

    # 4. 在全量训练集上重训练 SVM
    svm_final = SVC(kernel='rbf', C=0.85, gamma='scale')
    svm_final.fit(X_train, y_train)

    # 5. 提取测试集特征 & 预测
    test_ds     = RPSFeatDataset('test.pkl', transform=transform_feat)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    cnn.eval()
    feats_test = []
    ids_all    = []
    with torch.no_grad():
        for x, ids in test_loader:
            x = x.to(device)
            f = cnn.extract(x)
            feats_test.append(f.cpu().numpy())
            ids_all.append(ids.numpy())
    X_test = np.vstack(feats_test)
    ids_all = np.hstack(ids_all)
    y_test_pred = svm_final.predict(X_test)

    # 6. 生成 submission
    submission = pd.DataFrame({'id': ids_all, 'label': y_test_pred})
    submission.to_csv('submission_noaug_svm.csv', index=False)
    print("✅ 已生成 submission_noaug_svm.csv")

if __name__ == '__main__':
    main()
