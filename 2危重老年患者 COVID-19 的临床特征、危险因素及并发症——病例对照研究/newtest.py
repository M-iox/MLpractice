import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

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

df=train_data


age_mapping = {'60-70': 1, '71-80': 2, '>80': 3}
df['Age'] = df['Age'].map(age_mapping)

# 特征和目标变量的划分
X = df.drop(labels=['Sl 2','outcome'], axis=1)
y = df['outcome']

# 创建KFold对象，这里使用5折交叉验证
kf = KFold(n_splits=5, random_state=42, shuffle=True)

# 创建随机森林回归模型
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    criterion='squared_error',  # 这里更正为'squared_error'
    random_state=42
)

# 使用cross_val_score进行k折交叉验证
scores = cross_val_score(rf, X, y, cv=kf, scoring='neg_mean_squared_error')

# 计算交叉验证结果的均值和标准差
mse_scores = -scores  # 转换为正值，因为MSE是正值
mean_mse = mse_scores.mean()
std_mse = mse_scores.std()

print(f"Mean Squared Error (MSE): {mean_mse}")
print(f"Standard Deviation of MSE: {std_mse}")

# 训练模型并输出特征重要性
rf.fit(X, y)  # 使用整个数据集训练模型


feature_importances = rf.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("Feature Importances:")
print(importance_df)