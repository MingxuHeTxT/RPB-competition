import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score

# 1. 加载训练数据
with open('train.pkl', 'rb') as f:
    train = pickle.load(f)
imgs1 = np.stack(train['img1']).reshape(len(train['img1']), -1)
imgs2 = np.stack(train['img2']).reshape(len(train['img2']), -1)
X_all = np.hstack([imgs1, imgs2])   # (40000, 1152)
y_all = np.array(train['label'])    # (40000,)

# 2. 抽取 50% 的数据子集
X_sub, _, y_sub, _ = train_test_split(
    X_all, y_all,
    test_size=0.5,              # 只保留50%
    random_state=42,
    stratify=y_all
)
print(f"训练样本数（子集）: {X_sub.shape[0]}")

# 3. 手动做 3 折交叉验证
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
val_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_sub, y_sub)):
    print(f"🔵 第 {fold+1} 折：训练 {len(train_idx)}，验证 {len(val_idx)}")
    
    X_train, X_val = X_sub[train_idx], X_sub[val_idx]
    y_train, y_val = y_sub[train_idx], y_sub[val_idx]
    
    # 每一折重新初始化模型
    svm_clf = SVC(kernel='rbf', C=0.85, gamma='scale', verbose=False)
    svm_clf.fit(X_train, y_train)
    
    y_pred_val = svm_clf.predict(X_val)
    val_acc = accuracy_score(y_val, y_pred_val)
    val_scores.append(val_acc)
    
    print(f"   验证集准确率: {val_acc:.4f}")

# 4. 打印最终3折平均得分
avg_val_score = np.mean(val_scores)
print(f"\n✅ 3折交叉验证平均准确率: {avg_val_score:.4f}")
