# 导入必要的库
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
import ace_tools as tools

# 读取训练集和测试集数据
train_file_path = '/mnt/data/train1.csv'
test_file_path = '/mnt/data/test1.csv'

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# 数据预处理
# 将特征和目标变量分开
X = train_data.drop(columns=['Patient Code', 'Deceased'])
y = train_data['Deceased'].map({'No': 0, 'Yes': 1})  # 将目标变量映射为数值

# 检查训练集中数据类型为非数值的列
non_numeric_columns = X.select_dtypes(include=['object']).columns

# 数值特征
numeric_features = X.columns.drop(non_numeric_columns)

# 数值特征处理流水线
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # 使用均值填补缺失值
    ('scaler', StandardScaler())  # 标准化数据
])

# 非数值特征处理流水线
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # 使用'missing'填补缺失值
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # 独热编码
])

# 将预处理应用于数值和非数值特征
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),  # 数值特征
        ('cat', categorical_transformer, non_numeric_columns)  # 非数值特征
    ])

# 创建随机森林分类器模型
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
])

# 拆分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 在验证集上预测并计算AUC得分
y_val_pred_proba = model.predict_proba(X_val)[:, 1]
auc_score = roc_auc_score(y_val, y_val_pred_proba)

# 在测试集上进行预测
X_test = test_data.drop(columns=['Patient Code'])
y_test_pred_proba = model.predict_proba(X_test)[:, 1]

# 将结果保存到DataFrame中
prediction_results = pd.DataFrame({'id': test_data['Patient Code'], 'label': y_test_pred_proba})

# 显示预测结果
tools.display_dataframe_to_user(name="Prediction Results", dataframe=prediction_results)

# 打印验证集上的AUC分数
print(f"Validation AUC Score: {auc_score}")
