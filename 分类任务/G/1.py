import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import ace_tools as tools

# 读取训练集和测试集数据
train_data_path = '/mnt/data/train.csv'
test_data_path = '/mnt/data/test.csv'
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# 修正训练集列名中的空格问题
train_data.rename(columns=lambda x: x.strip(), inplace=True)

# 将 Troponin 列中的非数值数据转换为数值类型，'Negative' 转换为 0
train_data['Troponin'] = train_data['Troponin'].apply(lambda x: 0 if x == 'Negative' else float(x) if isinstance(x, str) and x.replace('.', '', 1).isdigit() else x)

# 数据预处理：提取特征和标签
X_train = train_data.drop(columns=['Patient Code', 'Group'])  # 去掉患者编码和标签
y_train = train_data['Group'].apply(lambda x: 1 if x != 'Control' else 0)  # 转换标签为0和1

# 构建数据处理和模型的pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # 使用均值填补缺失值
    ('scaler', StandardScaler()),                # 数据标准化
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))  # 随机森林分类器
])

# 训练模型
pipeline.fit(X_train, y_train)

# 对测试集进行相同的处理：转换 Troponin 列，并进行预测
test_data['Troponin'] = test_data['Troponin'].apply(lambda x: 0 if x == 'Negative' else float(x) if isinstance(x, str) and x.replace('.', '', 1).isdigit() else x)
X_test = test_data.drop(columns=['Patient Code'])  # 去掉患者编码

y_pred = pipeline.predict(X_test)

# 创建提交的DataFrame
submission = pd.DataFrame({'id': test_data['Patient Code'], 'Group': y_pred})

# 显示预测结果
tools.display_dataframe_to_user(name="预测结果", dataframe=submission)
