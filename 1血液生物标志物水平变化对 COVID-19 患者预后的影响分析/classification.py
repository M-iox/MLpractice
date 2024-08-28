from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

train_data = pd.read_excel('训练集.xlsx',sheet_name=0)
print(f'train_data.shape:{train_data.shape}')
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
all_features = train_data.iloc[:, 1:-1]

# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / x.std())
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 处理离散值
# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
all_features = pd.get_dummies(all_features, dummy_na=True, dtype=int)
print(f'all_features.shape: {all_features.shape}')
print(all_features.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])


n_train = train_data.shape[0]
train_data.columns = train_data.columns.str.strip()
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['Control', 'COVID-19'])
train_labels = label_encoder.transform(train_data['Group'].values.ravel())
print(train_labels)