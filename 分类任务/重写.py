from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
class Model:
    def __init__(self):
        self.train_data = pd.read_csv('train.csv')
        self.test_data = pd.read_csv('test.csv')
        self.train_features = self.process_data(self.train_data, mode='train')
        self.test_features  = self.process_data(self.test_data, mode='test')

    def process_data(self, data, mode):
        data.columns = data.columns.str.strip()  # 去掉列名的空格
        data = data.drop(columns=['IL-6'])
        # 处理数据
        if mode == 'train':
            # 训练数据处理逻辑
            # 处理标签
            self.labels = data['Group']
            label_encoder = LabelEncoder()
            self.labels = label_encoder.fit_transform(self.labels)
            self.labels = np.where(self.labels == 0, 1, 0)

            data = data.iloc[:, 2:-1]

        elif mode == 'test':
            # 测试数据处理逻辑
            data = data.iloc[:, 1:-1]

        # 处理缺失值
        numeric_features = data.dtypes[data.dtypes != 'object'].index
        categorical_features = data.select_dtypes(include=['object']).columns

        numeric_imputer = SimpleImputer(strategy='median')
        categorical_imputer = SimpleImputer(strategy='most_frequent')

        data[numeric_features] = numeric_imputer.fit_transform(data[numeric_features])
        data[categorical_features] = categorical_imputer.fit_transform(data[categorical_features])

        # One-Hot编码处理离散值
        data = pd.get_dummies(data, dummy_na=True, dtype=int)
        return data

model = Model()

print(model.train_features)
print(model.test_features)
print(model.labels)