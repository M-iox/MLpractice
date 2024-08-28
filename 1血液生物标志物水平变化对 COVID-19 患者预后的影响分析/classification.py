from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
train_data = pd.read_excel('训练集.xlsx', sheet_name=0)
train_data.columns = train_data.columns.str.strip()

# 提取特征
all_features = train_data.iloc[:, 2:-1]

# 填充NA值
## 确定数值型特征
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index

## 对数值型特征用中位数填充
numeric_imputer = SimpleImputer(strategy='median')
all_features[numeric_features] = numeric_imputer.fit_transform(all_features[numeric_features])

## 确定分类特征并进行众数填充
categorical_features = all_features.columns.difference(numeric_features)
categorical_imputer = SimpleImputer(strategy='most_frequent')
all_features[categorical_features] = categorical_imputer.fit_transform(all_features[categorical_features])

## 处理离散值（在填充后进行）
all_features = pd.get_dummies(all_features, dummy_na=True, dtype=int)

# 处理标签
n_train = train_data.shape[0]
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['Control', 'COVID-19'])
train_labels = label_encoder.transform(train_data['Group'].values.ravel())

#标准化 (如果需要的话，可以在这里添加标准化步骤)

def custom_scorer(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    return 0.5 * acc + 0.5 * f1

scorer = make_scorer(custom_scorer)

# 模型训练和交叉验证
X_train = all_features
y_train = train_labels
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# k折交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)
accuracies = []
f1_scores = []
custom_scores = []

for train_index, test_index in kf.split(X_train):
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_fold, y_train_fold)

    y_pred_fold = rf_model.predict(X_test_fold)

    acc = accuracy_score(y_test_fold, y_pred_fold)
    f1 = f1_score(y_test_fold, y_pred_fold, average='weighted')
    custom = custom_scorer(y_test_fold, y_pred_fold)

    accuracies.append(acc)
    f1_scores.append(f1)
    custom_scores.append(custom)

# 输出各个指标的交叉验证结果
print(f'Accuracy scores: {accuracies}')
print(f'Mean accuracy: {np.mean(accuracies):.4f}')
print(f'Standard deviation: {np.std(accuracies):.4f}')

print(f'F1 scores: {f1_scores}')
print(f'Mean F1 score: {np.mean(f1_scores):.4f}')
print(f'Standard deviation: {np.std(f1_scores):.4f}')

print(f'Custom scores (0.5*Acc + 0.5*F1): {custom_scores}')
print(f'Mean custom score: {np.mean(custom_scores):.4f}')
print(f'Standard deviation: {np.std(custom_scores):.4f}')

# 绘制每一折的准确率、F1分数和自定义评分的柱状图
plt.figure(figsize=(15, 6))

# 准确率
plt.subplot(1, 3, 1)
plt.bar(range(1, len(accuracies) + 1), accuracies, color='blue', alpha=0.7)
plt.xlabel('Fold Number')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Accuracy for Each Fold')
plt.ylim(0.8, 1.0)
for i in range(len(accuracies)):
    plt.text(i + 1, accuracies[i], f'{accuracies[i]:.4f}', ha='center', va='bottom')

# F1分数
plt.subplot(1, 3, 2)
plt.bar(range(1, len(f1_scores) + 1), f1_scores, color='green', alpha=0.7)
plt.xlabel('Fold Number')
plt.ylabel('F1 Score')
plt.title('Cross-Validation F1 Score for Each Fold')
plt.ylim(0.8, 1.0)
for i in range(len(f1_scores)):
    plt.text(i + 1, f1_scores[i], f'{f1_scores[i]:.4f}', ha='center', va='bottom')

# 自定义评分
plt.subplot(1, 3, 3)
plt.bar(range(1, len(custom_scores) + 1), custom_scores, color='purple', alpha=0.7)
plt.xlabel('Fold Number')
plt.ylabel('Custom Score')
plt.title('Cross-Validation Custom Score for Each Fold')
plt.ylim(0.8, 1.0)
for i in range(len(custom_scores)):
    plt.text(i + 1, custom_scores[i], f'{custom_scores[i]:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()
# 0.9923,0.0032
# 0.9934 0.0064