from classification import RandomForestModel
from regression import RandomForestRegressionModel
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
def process_test_data(test_data_path, sheet_name=0):
    # 读取测试集数据
    test_data = pd.read_excel(test_data_path, sheet_name=sheet_name)
    test_data.columns = test_data.columns.str.strip()

    # 提取特征数据
    test_features = test_data.iloc[:, 2:-1]

    # 填充数值型和分类特征的NA值
    numeric_features = test_features.dtypes[test_features.dtypes != 'object'].index
    numeric_imputer = SimpleImputer(strategy='median')
    test_features[numeric_features] = numeric_imputer.fit_transform(test_features[numeric_features])

    categorical_features = test_features.columns.difference(numeric_features)
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    test_features[categorical_features] = categorical_imputer.fit_transform(test_features[categorical_features])

    # One-Hot编码处理离散值
    test_features = pd.get_dummies(test_features, dummy_na=True, dtype=int)

    return test_features

# 设置显示所有行和列
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# 设置不折叠长列表或数组
np.set_printoptions(threshold=np.inf)

# 使用函数处理测试集
test_features = process_test_data('训练集.xlsx')

ClassficationModel = RandomForestModel(data_path='训练集.xlsx')
ClassficationModel.default_parameters()
ClassficationModel.cross_validate()
predictions, patient_index = ClassficationModel.predict_full_dataset(test_features)

print("患者预测结果",predictions)
if len(patient_index) > 0:
    # 提取分类为1的样本的特征
    high_risk_features = test_features.iloc[patient_index]

    # 使用回归模型预测死亡率
    RegressionModel = RandomForestRegressionModel(data_path='训练集.xlsx')
    RegressionModel.default_parameters()
    RegressionModel.cross_validate()
    mortality_predictions = RegressionModel.predict_full_dataset(high_risk_features)

    print("高风险病人的下标和死亡率: ",patient_index, mortality_predictions)
else:
    print("没有找到高风险病人.")