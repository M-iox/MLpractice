from classification import RandomForestModel
from regression import RandomForestRegressionModel
import pandas as pd
from sklearn.impute import SimpleImputer

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

# 使用函数处理测试集
test_features = process_test_data('训练集.xlsx')

ClassficationModel = RandomForestModel(data_path='训练集.xlsx')
ClassficationModel.default_parameters()
ClassficationModel.cross_validate()

print(f"ClassficationModel.predict_full_dataset():{ClassficationModel.predict_full_dataset(test_features)}")


