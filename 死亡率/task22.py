from sklearn.feature_selection import SelectFromModel
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
class model:
    def __init__(self):
        # 加载训练集数据
        self.file_path = './train2.xlsx'
        self.train_data = pd.read_excel(self.file_path)
        self.X_train = self.train_data.drop(columns=['Sl 2', 'outcome'])
        self.y_train = self.train_data['outcome'].apply(lambda x: 1 if x == 2 else 0)

        # 加载测试数据
        self.test_file_path = './test2.csv'
        self.test_data_processed = pd.read_csv(self.test_file_path)
        self.id_number = self.test_data_processed['Patient Code']
        self.test_data_processed = self.test_data_processed.drop(columns=['Patient Code'])

        # 去掉空格
        self.X_train.columns = self.X_train.columns.str.strip()
        self.test_data_processed.columns = self.test_data_processed.columns.str.strip()

        # 初始化scaler和pca
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)

        # 数据处理
        self.X_train = self.data_process(self.X_train)
        self.test_data_processed = self.data_process(self.test_data_processed)

        # 独热编码的去重和对齐
        self.X_train = self.X_train.loc[:, ~self.X_train.columns.duplicated()]
        self.test_data_processed = self.test_data_processed.loc[:, ~self.test_data_processed.columns.duplicated()]
        self.test_data_processed = self.test_data_processed.reindex(columns=self.X_train.columns, fill_value=0)


        # 特征选择
        self.X_train, self.test_data_processed = self.select_important_features()
        print(self.X_train.shape)

        # pca和标准化
        self.X_train = self.standardize_and_reduce(self.X_train, fit=True)
        self.test_data_processed = self.standardize_and_reduce(self.test_data_processed, fit=False)

        # 训练和预测
        self.train_test()

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


    def data_process(self, X_train):
        # 大部分为int的列
        empty = ['Procalcitonin (ng/mL)', 'IL-6 (pg/mL)']
        X_train = X_train.drop(columns=empty)

        all_columns = X_train.columns.tolist()
        int_columns = ['Heart Rate', 'RR', 'Systolic Blood Pressure', 'Diastolic Blood Pressure', 'spO2 at Room air',
                       'WBC count (x109/L)', 'Lymphocyte count (109/L)', 'Neutrophil count (109/L)',
                       'HbA1c (%)', 'Creatinine (μmol/L)', 'Sodium (mmol/L)', 'Potassium (mmol/L)',
                       'CRP (mg/L)', 'LDH (U/L)', 'D-dimer (mg/L)', 'Ferritin (ng/mL)']
        # 标签处理或独热的列

        str_columns = [col for col in all_columns if col not in int_columns]
        # 将str类型转化为空值

        for col in int_columns:
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce')

        # 处理空值
        numeric_imputer = SimpleImputer(strategy='median')
        X_train[int_columns] = numeric_imputer.fit_transform(X_train[int_columns])


        # 转化为str
        for col in str_columns:
            X_train[col] = X_train[col].astype(str)

        # 替换空值为‘0’
        X_train = X_train[str_columns].replace({'NA': '0'})

        # 处理分类变量和缺失值
        X_train = pd.get_dummies(X_train, columns=str_columns, dummy_na=True, dtype=int, drop_first=True).fillna(0)

        return X_train

    def select_important_features(self):
        # 使用随机森林进行特征选择
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.X_train, self.y_train)

        # 使用SelectFromModel选择最重要的特征
        selector = SelectFromModel(rf, threshold=0.0044, prefit=True)
        X_train_selected = selector.transform(self.X_train)
        test_data_selected = selector.transform(self.test_data_processed)

        # 打印选定的特征
        selected_features = self.X_train.columns[selector.get_support()]
        print(f"Selected features: {selected_features}")

        return pd.DataFrame(X_train_selected), pd.DataFrame(test_data_selected)

    def tuned_para(self):
        # 定义 SVM 的参数搜索范围
        param_grid = {
            'C': [0.1, 1, 10, 100, 1000],  # 惩罚系数
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],  # 核函数系数
            'kernel': ['rbf', 'poly', 'sigmoid'],  # 核函数类型
            'degree': [3, 4, 5]  # 仅对多项式核有效
        }

        svm = SVC(probability=True, random_state=42)

        # 使用 GridSearchCV 进行超参数调优
        grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=1, verbose=2)
        grid_search.fit(self.X_train, self.y_train)

        print(f"Best parameters found: {grid_search.best_params_}")
        self._model = grid_search.best_estimator_

    def standardize_and_reduce(self, data, fit=True):
        if fit:
            # 拟合标准化器和PCA，并转换数据
            data = self.scaler.fit_transform(data)
            data = self.pca.fit_transform(data)
        else:
            # 使用已拟合的标准化器和PCA转换数据
            data = self.scaler.transform(data)
            data = self.pca.transform(data)
        return pd.DataFrame(data)

    def train_test(self, n_splits=5):
        # 使用K折交叉验证
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        train_maes = []
        valid_maes = []
        train_rmses = []
        valid_rmses = []
        train_custom_scores = []
        valid_custom_scores = []

        # 对每个折进行训练和验证
        for fold, (train_index, valid_index) in enumerate(skf.split(self.X_train, self.y_train), 1):
            # 使用.iloc基于位置索引来提取训练集和验证集
            X_train_split, X_valid = self.X_train.iloc[train_index], self.X_train.iloc[valid_index]
            y_train_split, y_valid = self.y_train.iloc[train_index], self.y_train.iloc[valid_index]

            self._model = SVC(kernel='rbf', C=1, gamma=0.1, degree=3, random_state=42)
            self._model.fit(X_train_split, y_train_split)

            self.calibrated_svm = CalibratedClassifierCV(estimator=self._model, method='sigmoid', cv='prefit')
            self.calibrated_svm.fit(X_train_split, y_train_split)

            # 预测验证集
            y_valid_pred = self.calibrated_svm.predict_proba(X_valid)[:, 1]
            y_train_pred = self.calibrated_svm.predict_proba(X_train_split)[:, 1]

            # 计算误差和自定义评分
            train_mae = mean_absolute_error(y_train_split, y_train_pred)
            valid_mae = mean_absolute_error(y_valid, y_valid_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train_split, y_train_pred))
            valid_rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred))

            train_custom_score = self.custom_scorer(y_train_split, y_train_pred)
            valid_custom_score = self.custom_scorer(y_valid, y_valid_pred)

            train_maes.append(train_mae)
            valid_maes.append(valid_mae)
            train_rmses.append(train_rmse)
            valid_rmses.append(valid_rmse)
            train_custom_scores.append(train_custom_score)
            valid_custom_scores.append(valid_custom_score)

            print(f"Fold {fold}:")
            self._print_results(valid_mae, valid_rmse, [valid_custom_score])

        # 输出K折验证的平均结果
        print("\nOverall Results:")
        self._print_results(np.mean(valid_maes), np.mean(valid_rmses), valid_custom_scores)

        # 绘制对比图
        self._plot_comparison(train_maes, valid_maes, train_rmses, valid_rmses, train_custom_scores,
                              valid_custom_scores)

        # 在整个训练集上训练模型并在测试集上进行预测
        self._model.fit(self.X_train, self.y_train)
        self.calibrated_svm = CalibratedClassifierCV(estimator=self._model, method='sigmoid', cv='prefit')
        self.calibrated_svm.fit(self.X_train, self.y_train)

        y_test_pred = self.calibrated_svm.predict_proba(self.test_data_processed)[:, 1]

        # 准备预测结果
        predictions = pd.DataFrame({
            'id': self.id_number,
            'label': y_test_pred
        })

        # 将预测结果保存为CSV文件
        output_file_path = './result2.csv'
        predictions.to_csv(output_file_path, index=False)


Model = model()
