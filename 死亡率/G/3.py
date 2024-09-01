# 导入必要的库
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# 加载训练集数据
file_path = '/mnt/data/train3.xlsx'
train_data = pd.read_excel(file_path)

# 准备训练数据
X_train = train_data.drop(columns=['Number', 'Death (1 Yes 2 No)'])
y_train = train_data['Death (1 Yes 2 No)'].apply(lambda x: 1 if x == 1 else 0)

# 处理分类变量和缺失值
X_train = pd.get_dummies(X_train, drop_first=True).fillna(0)

# 加载测试集数据
test_file_path = '/mnt/data/test3.csv'
test_data = pd.read_csv(test_file_path, encoding='ISO-8859-1')

# 处理测试集数据
test_data_processed = pd.get_dummies(test_data.drop(columns=['Patient Code']), drop_first=True).fillna(0)

# 确保测试集和训练集的数据列对齐
test_data_processed = test_data_processed.reindex(columns=X_train.columns, fill_value=0)

# 将训练数据分割为训练集和验证集
X_train_split, X_valid, y_train_split, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 训练模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_split, y_train_split)

# 验证模型
y_valid_pred = model.predict(X_valid)
validation_error = mean_squared_error(y_valid, y_valid_pred)
validation_error_rmse = np.sqrt(validation_error)

# 在测试集上进行预测
y_test_pred = model.predict(test_data_processed)

# 准备预测结果
predictions = pd.DataFrame({
    'id': test_data['Patient Code'],
    'label': y_test_pred
})

# 将预测结果保存为CSV文件
output_file_path = '/mnt/data/predictions.csv'
predictions.to_csv(output_file_path, index=False)

# 输出结果文件路径
output_file_path
