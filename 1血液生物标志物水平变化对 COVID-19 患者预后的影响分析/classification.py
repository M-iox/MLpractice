from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

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

