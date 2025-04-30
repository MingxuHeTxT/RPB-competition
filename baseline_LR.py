import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# 1) 加载全部训练数据
with open('train.pkl','rb') as f:
    train = pickle.load(f)
imgs1 = np.stack(train['img1']).reshape(len(train['img1']), -1)
imgs2 = np.stack(train['img2']).reshape(len(train['img2']), -1)
X_all = np.hstack([imgs1, imgs2])        # (40000, 1152)
y_all = np.array(train['label'])

# 2) 用“最佳超参数”重训练模型（这里假设 LogisticRegression, C=0.1）
best_clf = LogisticRegression(C=0.1, max_iter=1000, n_jobs=-1)
best_clf.fit(X_all, y_all)

from sklearn.metrics import accuracy_score

y_pred_train = best_clf.predict(X_all)
train_acc = accuracy_score(y_all, y_pred_train)
print(f"训练集上的准确率（仅供参考）: {train_acc:.4f}")

# 3) 加载测试集并预测
with open('test.pkl','rb') as f:
    test = pickle.load(f)
imgs1_test = np.stack(test['img1']).reshape(len(test['img1']), -1)
imgs2_test = np.stack(test['img2']).reshape(len(test['img2']), -1)
X_test     = np.hstack([imgs1_test, imgs2_test])
ids        = test['id']

y_test_pred = best_clf.predict(X_test)

# 4) 生成 submission.csv
sub = pd.DataFrame({"id": ids, "label": y_test_pred})
sub.to_csv("submission.csv", index=False)
print("✅ submission.csv 已生成，上传即可查看成绩。")
