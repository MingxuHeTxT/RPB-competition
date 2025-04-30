import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# ═════════════════════════════════════════
# 1. 加载数据
# ═════════════════════════════════════════
with open('train.pkl', 'rb') as f:
    train = pickle.load(f)
imgs1 = np.stack(train['img1']).reshape(len(train['img1']), -1)
imgs2 = np.stack(train['img2']).reshape(len(train['img2']), -1)
X_all = np.hstack([imgs1, imgs2])
y_all = np.array(train['label'])

# 取10%子集
X_sub, _, y_sub, _ = train_test_split(
    X_all, y_all,
    test_size=0.9,
    random_state=42,
    stratify=y_all
)
print(f"🔹 子集大小: {X_sub.shape}")

# ═════════════════════════════════════════
# 2. Logistic Regression + GridSearchCV
# ═════════════════════════════════════════
print("🔍 LogisticRegression GridSearchCV...")

param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100]
}

grid_search_lr = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000, solver='lbfgs'),  # 用 lbfgs 最常规
    param_grid=param_grid_lr,
    cv=3,
    verbose=2,
    n_jobs=-1
)

grid_search_lr.fit(X_sub, y_sub)

print("✅ LogisticRegression 最佳参数:", grid_search_lr.best_params_)
print("✅ LogisticRegression 最佳交叉验证得分:", grid_search_lr.best_score_)

# ═════════════════════════════════════════
# 3. SVM (RBF Kernel) + GridSearchCV
# ═════════════════════════════════════════
print("\n🔍 SVM(RBF) GridSearchCV...")

param_grid_svm = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.001]
}

grid_search_svm = GridSearchCV(
    estimator=SVC(kernel='rbf'),
    param_grid=param_grid_svm,
    cv=3,
    verbose=2,
    n_jobs=-1
)

grid_search_svm.fit(X_sub, y_sub)

print("✅ SVM 最佳参数:", grid_search_svm.best_params_)
print("✅ SVM 最佳交叉验证得分:", grid_search_svm.best_score_)

# ═════════════════════════════════════════
# 4. 打印最终对比
# ═════════════════════════════════════════
print("\n📊 最终公平比较结果：")
print(f"LogisticRegression 验证集得分: {grid_search_lr.best_score_:.4f}")
print(f"SVM(RBF)            验证集得分: {grid_search_svm.best_score_:.4f}")
