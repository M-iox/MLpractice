import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.model_selection import train_test_split, KFold
class Model:
    def __init__(self):
        self.train_data = pd.read_excel('train2.xlsx')
        self.test_data = pd.read_csv('test2.csv')
        self.train_features,self.test_features = self.process_data()

        self.default_para()

        self.train_test(self.train_features,'train')
        self.train_test(self.test_features, 'test')

    def _print_results(self, mae, rmse, custom_scores):
        print(f'Custom scores (0.5 * mae + 0.5 * rmse): {custom_scores}')
        print(f'Mean custom score: {np.mean(custom_scores):.4f}')
        print(f'Standard deviation: {np.std(custom_scores):.4f}')

    def custom_scorer(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return 0.5 * mae + 0.5 * rmse

    def _plot_comparison(self, train_maes, valid_maes, train_rmses, valid_rmses , train_custom_scores, valid_custom_scores):
        plt.figure(figsize=(18, 12))

        # Custom Score 对比
        plt.subplot(2, 2, 4)
        plt.plot(train_custom_scores, label='Train Custom Score', marker='o', color='purple')
        plt.plot(valid_custom_scores, label='Validation Custom Score', marker='o', color='pink')
        plt.xlabel('Fold Number')
        plt.ylabel('Custom Score')
        plt.title('Train vs Validation Custom Score')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def process_data(self):
        self.train_features = self.train_data.copy()
        self.test_features = self.test_data.copy()

        # 去掉列名的空格
        self.train_features.rename(columns=lambda x: x.strip(), inplace=True)
        self.test_features.rename(columns=lambda x: x.strip(), inplace=True)

        self.train_features['Age']=self.train_features['Age'].map({'60-70': 1, '71-80': 2, '>80': 3})
        self.test_features['Age'] = self.train_features['Age'].map({'60-70': 1, '71-80': 2, '>80': 3})

        # 数据预处理：提取特征和标签
        self.labels = self.train_features['outcome'].apply(lambda x: 1 if x == 2 else 0)  # 转换标签为0和1
        self.train_features = self.train_features.drop(columns=['Sl 2', 'outcome'])  # 去掉患者编码和标签
        print(f"self.labels:{self.labels}")

        # 测试集去掉编码
        self.test_features =self.test_features.drop(columns=['Patient Code'])  # 去掉患者编码

        return self.train_features , self.test_features

    def default_para(self):
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),  # 使用均值填补缺失值
            ('scaler', StandardScaler()),  # 数据标准化
            ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))  # 随机森林分类器
        ])
        self._model = pipeline

    def tuned_para(self):
        self.default_para()
        pass

    def train_test(self, data, mode):
        if mode == 'train':
            kf = KFold(n_splits=7, shuffle=True, random_state=42)
            train_maes, valid_maes = [], []
            train_rmses, valid_rmses = [], []
            train_custom_scores, valid_custom_scores = [], []

            for train_index, valid_index in kf.split(data):
                X_train_fold, X_valid_fold = data.iloc[train_index], data.iloc[valid_index]
                y_train_fold, y_valid_fold = self.labels[train_index], self.labels[valid_index]

                self._model.fit(X_train_fold, y_train_fold)

                # 预测训练集和验证集
                y_train_pred_fold = self._model.predict(X_train_fold)
                y_valid_pred_fold = self._model.predict(X_valid_fold)

                # 计算训练集和验证集的各项指标
                train_mae = mean_absolute_error(y_train_fold, y_train_pred_fold)
                valid_mae = mean_absolute_error(y_valid_fold, y_valid_pred_fold)

                train_rmse = np.sqrt(mean_squared_error(y_train_fold, y_train_pred_fold))
                valid_rmse = np.sqrt(mean_squared_error(y_valid_fold, y_valid_pred_fold))

                train_custom_score = self.custom_scorer(y_train_fold, y_train_pred_fold)
                valid_custom_score = self.custom_scorer(y_valid_fold, y_valid_pred_fold)

                # 记录每个 fold 的结果
                train_maes.append(train_mae)
                valid_maes.append(valid_mae)
                train_rmses.append(train_rmse)
                valid_rmses.append(valid_rmse)
                train_custom_scores.append(train_custom_score)
                valid_custom_scores.append(valid_custom_score)

            # 打印并绘制训练集和验证集的结果
            self._plot_comparison(train_maes, valid_maes, train_rmses, valid_rmses, train_custom_scores, valid_custom_scores)
            self._print_results(valid_maes, valid_rmses, valid_custom_scores)

        if mode == 'test':
            predictions_proba = self._model.predict_proba(data)[:, 1]
            results = pd.DataFrame({'id': self.test_data.iloc[:, 0], 'Probability': predictions_proba})
            output_csv = 'result2.csv'
            results.to_csv(output_csv, index=False)

model = Model()
