import pickle
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

# 1) 加载全部训练数据
with open('train.pkl', 'rb') as f:
    train = pickle.load(f)
imgs1 = np.stack(train['img1']).reshape(len(train['img1']), -1)
imgs2 = np.stack(train['img2']).reshape(len(train['img2']), -1)
X_all = np.hstack([imgs1, imgs2])   # (40000, 1152)
y_all = np.array(train['label'])    # (40000,)

# 2) 抽取 10% 子集用于调参
X_sub, _, y_sub, _ = train_test_split(
    X_all, y_all,
    test_size=0.9,
    random_state=42,
    stratify=y_all
)
print(f"🔹 子集大小: {X_sub.shape}")

# 3) 扩大参数搜索范围
param_grid = {
    'C':     [ 0.83, 0.84, 0.845, 0.85, 0.855, 0.86, 0.87],
    'gamma': ['scale']
}

grid_search = GridSearchCV(
    SVC(kernel='rbf'),
    param_grid,
    cv=3,
    verbose=2,
    n_jobs=-1
)

print("🔍 开始 GridSearchCV 调参 …")
grid_search.fit(X_sub, y_sub)
print("✅ 超参数搜索完成")
print("最佳参数     :", grid_search.best_params_)
print("最佳交叉验证得分:", grid_search.best_score_)


# —— 到此为止，testing 阶段完成，暂不生成 submission.csv ——  
