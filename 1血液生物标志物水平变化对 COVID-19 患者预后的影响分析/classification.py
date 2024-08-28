from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
train_data = pd.read_excel('训练集.xlsx', sheet_name=0)
train_data.columns = train_data.columns.str.strip()
print(f'train_data.shape:{train_data.shape}')
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

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
print(f'all_features.shape: {all_features.shape}')
print(all_features.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

# 处理标签
n_train = train_data.shape[0]
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['Control', 'COVID-19'])
train_labels = label_encoder.transform(train_data['Group'].values.ravel())
print(train_labels)

#标准化 (如果需要的话，可以在这里添加标准化步骤)

# 模型训练和交叉验证
X_train = all_features
y_train = train_labels
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# k折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=kf, scoring='accuracy')

# 绘制每一折的准确性得分的柱状图
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(cv_scores) + 1), cv_scores, color='blue', alpha=0.7)
plt.xlabel('Fold Number')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Accuracy for Each Fold')
plt.ylim(0.8, 1.0)  # 你可以根据具体情况调整y轴范围

# 在每个柱上标注准确性得分
for i in range(len(cv_scores)):
    plt.text(i + 1, cv_scores[i], f'{cv_scores[i]:.4f}', ha='center', va='bottom')

plt.show()
# 输出交叉验证的结果
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean accuracy: {cv_scores.mean():.4f}')
print(f'Standard deviation: {cv_scores.std():.4f}')

# 如果需要训练最终模型
rf_model.fit(X_train, y_train)

# 输出分类报告
y_pred = rf_model.predict(X_train)
print(classification_report(y_train, y_pred))
