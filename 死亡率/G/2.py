import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

# 读取训练集数据
file_path = '/mnt/data/train2.xlsx'
train_data = pd.read_excel(file_path, sheet_name='Sheet1')

# 删除无关列
train_data = train_data.drop(columns=['Sl 2'])

# 提取目标变量和特征
target = train_data['outcome']
features = train_data.drop(columns=['outcome'])

# 编码分类数据
label_encoders = {}
for column in features.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    features[column] = le.fit_transform(features[column].astype(str))
    label_encoders[column] = le

# 处理缺失值
imputer = SimpleImputer(strategy='mean')
features_imputed = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)

# 划分数据集
X_train, X_val, y_train, y_val = train_test_split(features_imputed, target, test_size=0.2, random_state=42)

# 训练随机森林模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 评估模型性能
y_pred_val = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred_val)
print(f"Mean Squared Error on validation set: {mse}")

# 读取测试集数据
test_file_path = '/mnt/data/test2.csv'
test_data = pd.read_csv(test_file_path)

# 对测试集中的分类数据进行编码
test_features = test_data.drop(columns=['Patient Code'])
for column, le in label_encoders.items():
    if column in test_features.columns:
        test_features[column] = le.transform(test_features[column].astype(str))

# 填充测试集中的缺失值
test_features_imputed = pd.DataFrame(imputer.transform(test_features), columns=test_features.columns)

# 预测测试集的死亡率
predictions = model.predict(test_features_imputed)

# 将预测值限制在0到1之间
predictions = predictions.clip(0, 1)

# 创建结果数据框
results = pd.DataFrame({'id': test_data['Patient Code'], 'label': predictions})

# 显示结果数据框
import ace_tools as tools; tools.display_dataframe_to_user(name="Predicted Mortality Rates", dataframe=results)
