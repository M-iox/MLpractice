from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, \
    RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor

class Model:
    def __init__(self):
        self.train_data = {}  # 使用字典来存储数据
        self.test_data = {}
        self.train_features = {}
        self.test_features = {}
        self.labels = {}

        # 读取数据
        self.test_data[1] = pd.read_csv('test1.csv')
        self.test_data[2] = pd.read_csv('test2.csv')
        self.test_data[3] = pd.read_csv('test3.csv', encoding='latin1')

        self.train_data[1] = pd.read_csv('train1.csv')
        #self.train_data[1] = pd.read_excel('data1.xlsx', sheet_name='Sheet2')
        # train2 和 train3 是 Excel 文件
        self.train_data[2] = pd.read_excel('train2.xlsx', sheet_name='Sheet1')
        self.train_data[3] = pd.read_excel('train3.xlsx', sheet_name='Sheet1')

        self.train_features = {i: self.train_data[i].copy() for i in range(1, 4)}
        self.test_features = {i: self.test_data[i].copy() for i in range(1, 4)}
        self.labels = {1: 'label1', 2: 'label2', 3: 'label3'}

        self.process_data()

        self.default_para()
        self.train_and_predict()

    def process_label(self):
        # 将非数值标签转换为数值形式
        self.labels[1] = self.train_data[1]['Deceased'].map({'No': 0, 'Yes': 1})  # No存活，Yes死亡
        self.labels[2] = self.train_data[2]['outcome'].map({1: 0, 2: 1})  # 1表示存活, 2表示死亡
        self.labels[3] = self.train_data[3]['Death (1 Yes 2 No)'].map({1: 1, 2: 0})  # 1死亡，2存活
        for i in range(1,4):
            print(self.labels[i])

    def process_data(self):
        # 初始化独热编码所需的列
        self.encoder_columns = None
        # 初始化标准化和PCA对象
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # 保留95%的方差

        for i in range(1,4):
            # 去掉列名的空格
            self.train_features[i].columns = self.train_features[i].columns.str.strip()
            self.test_features[i].columns = self.test_features[i].columns.str.strip()

            # 将空字符串替换为 NaN
            self.train_features[i]= self.train_features[i].replace('', pd.NA)
            self.test_features[i]= self.test_features[i].replace('', pd.NA)

            # 1. 找出 test_data 中全为 NaN 的列
            na_columns = self.test_features[i].columns[self.test_features[i].isna().all()]
            # 2. 删除 train_data 和 test_data 中这些列
            self.train_features[i] = self.train_features[i].drop(columns=na_columns, errors='ignore')
            self.test_features[i] = self.test_features[i].drop(columns=na_columns, errors='ignore')


        # 训练数据处理逻辑
        # 处理标签
        self.process_label()

        self.train_features[1] = self.train_features[1].iloc[:,2:]
        self.train_features[2] = self.train_features[2].iloc[:, 2:]
        self.train_features[3] = self.train_features[3].iloc[:, 1:-2].join(self.train_features[3].iloc[:, -1])


        self.test_features[1] = self.test_features[1].iloc[:, 1:]
        self.test_features[2] = self.test_features[2].iloc[:, 1:]
        self.test_features[3] = self.test_features[3].iloc[:, 1:]

        # 处理缺失值、异常值和独热编码
        for i in range(1, 4):
            self.train_features[i] = self.handle_missing_and_encode(self.train_features[i], fit=True)
            self.test_features[i] = self.handle_missing_and_encode(self.test_features[i], fit=False)
            # 新增：处理异常值
            self.train_features[i] = self.handle_outliers(self.train_features[i])
            self.test_features[i] = self.handle_outliers(self.test_features[i])
            # 标准化和PCA降维
            self.train_features[i] = self.standardize_and_reduce(self.train_features[i], fit=True)
            # self.test_features[i] = self.standardize_and_reduce(self.test_features[i], fit=False)

        for i in range(1, 4):
            # print(i)
            self.test_features[i] = self.handle_missing_and_encode(self.test_features[i])

            # 独热编码的去重和对齐
            self.train_features[i] = self.train_features[i].loc[:, ~self.train_features[i].columns.duplicated()]
            self.test_features[i] = self.test_features[i].loc[:, ~self.test_features[i].columns.duplicated()]

            self.test_features[i] = self.test_features[i].reindex(columns=self.train_features[i].columns, fill_value=0)
    def handle_outliers(self, data, threshold=1.5):
        """
        使用IQR（四分位距）方法来处理数据中的异常值
        :param data: DataFrame, 需要处理的特征数据
        :param threshold: float, 用于判断异常值的IQR倍数阈值，通常为1.5
        :return: DataFrame, 处理后的数据
        """
        # 只处理数值型特征
        numeric_features = data.select_dtypes(include=[np.number])

        for column in numeric_features.columns:
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            # 将异常值截断为边界值
            data[column] = np.where(data[column] < lower_bound, lower_bound, data[column])
            data[column] = np.where(data[column] > upper_bound, upper_bound, data[column])

        return data
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

    def handle_missing_and_encode(self, data , fit=False):
        # 处理缺失值
        numeric_features = data.dtypes[data.dtypes != 'object'].index
        categorical_features = data.select_dtypes(include=['object']).columns

        numeric_imputer = SimpleImputer(strategy='median')
        categorical_imputer = SimpleImputer(strategy='most_frequent')

        # 检查是否有数值特征
        if not numeric_features.empty:
            data[numeric_features] = numeric_imputer.fit_transform(data[numeric_features])

        # 检查是否有分类特征
        if not categorical_features.empty:
            # 将分类特征转换为字符串类型，避免类型冲突
            data[categorical_features] = data[categorical_features].astype(str)
            data[categorical_features] = categorical_imputer.fit_transform(data[categorical_features])

        # 独热编码处理离散值
        data = pd.get_dummies(data, dummy_na=True, dtype=int)

        # 检查并移除重复列
        data = data.loc[:, ~data.columns.duplicated()]

        if fit:
            self.encoder_columns = data.columns  # 保存训练集中编码后的列名
        else:
            # 确保测试集的特征与训练集对齐
            data = data.reindex(columns=self.encoder_columns, fill_value=0)

        return data

    def ensemble_model(self):
        # 使用回归模型替代分类模型
        self.svr = SVR(kernel='rbf', C=1, gamma='scale')
        self.rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)
        self.gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
        self.et = ExtraTreesRegressor(n_estimators=100, random_state=42)
        self.svc1 = SVC(kernel='rbf', C=1, gamma='scale', probability=True, random_state=42)  # 使用RBF核函数
        self.svc2 = SVC(kernel='linear', C=0.1, probability=True, random_state=42)  # 使用线性核函数
        self.svc3 = SVC(kernel='poly', degree=3, C=0.5, gamma='scale', probability=True, random_state=42)  # 使用多项式核函数
    def default_para(self):
        self.ensemble_model()
        # # 修改为StackingRegressor或其他适合回归的集成方法
        # self._model = StackingRegressor(
        #     estimators=[
        #         ('rf', self.rf),
        #         ('gb', self.gb),
        #         ('et', self.et)
        #     ],
        #     final_estimator=GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42),
        #     cv=10,
        #     n_jobs=-1,
        #     passthrough=False
        # )
        self._model = self.svc2

    def tuned_para(self):
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'linear']
        }
        grid_search = GridSearchCV(estimator=SVR(), param_grid=param_grid, cv=10, n_jobs=-1, verbose=1)
        grid_search.fit(self.train_features, self.labels)  # 确保使用正确的特征和标签
        print("Best parameters found: ", grid_search.best_params_)
        self._model = SVR(**grid_search.best_params_)

    def svc_predict(self, data_order, data):
        self.calibrated_svm = CalibratedClassifierCV(estimator=self._model, method='sigmoid', cv='prefit')
        self.calibrated_svm.fit(self.train_features[data_order], self.labels[data_order])

        if data_order == 1 or 2:
            result = self.calibrated_svm.predict_proba(data)[:, 1]
        if data_order == 3:
            result = self.calibrated_svm.predict_proba(data)[:, 0]
        return result
    def train_and_predict(self):
        all_results = pd.DataFrame()

        for i in range(1, 4):
            X_train = self.train_features[i]
            y_train = self.labels[i]
            X_test = self.test_features[i]

            kf = KFold(n_splits=20, shuffle=True, random_state=42)
            train_maes, valid_maes = [], []
            train_rmses, valid_rmses = [], []
            train_r2s, valid_r2s = [], []
            train_custom_scores, valid_custom_scores = [], []

            for train_index, valid_index in kf.split(X_train):
                X_train_fold = X_train.iloc[train_index]
                X_valid_fold = X_train.iloc[valid_index]
                y_train_fold = y_train.iloc[train_index]
                y_valid_fold = y_train.iloc[valid_index]

                # 训练模型
                self._model.fit(X_train_fold, y_train_fold)

                # 在训练集和验证集上进行预测
                y_train_pred_fold = self.svc_predict(i, data = X_train_fold)
                y_valid_pred_fold = self.svc_predict(i, data = X_valid_fold)

                # 计算训练集和验证集的各项指标
                train_mae = mean_absolute_error(y_train_fold, y_train_pred_fold)
                valid_mae = mean_absolute_error(y_valid_fold, y_valid_pred_fold)

                train_rmse = np.sqrt(mean_squared_error(y_train_fold, y_train_pred_fold))
                valid_rmse = np.sqrt(mean_squared_error(y_valid_fold, y_valid_pred_fold))

                train_r2 = r2_score(y_train_fold, y_train_pred_fold)
                valid_r2 = r2_score(y_valid_fold, y_valid_pred_fold)

                train_custom_score = self.custom_scorer(y_train_fold, y_train_pred_fold)
                valid_custom_score = self.custom_scorer(y_valid_fold, y_valid_pred_fold)

                # 记录每个 fold 的结果
                train_maes.append(train_mae)
                valid_maes.append(valid_mae)
                train_rmses.append(train_rmse)
                valid_rmses.append(valid_rmse)
                train_r2s.append(train_r2)
                valid_r2s.append(valid_r2)
                train_custom_scores.append(train_custom_score)
                valid_custom_scores.append(valid_custom_score)

            # 打印和绘制训练集和验证集的结果对比
            self._plot_comparison(train_maes, valid_maes, train_rmses, valid_rmses, train_r2s, valid_r2s, train_custom_scores, valid_custom_scores)

            # 输出每个指标的均值和方差
            print(f"Dataset {i} - Valid MAE: Mean={np.mean(valid_maes):.4f}, Std={np.std(valid_maes):.4f}")
            print(f"Dataset {i} - Valid RMSE: Mean={np.mean(valid_rmses):.4f}, Std={np.std(valid_rmses):.4f}")
            print(f"Dataset {i} - Valid R²: Mean={np.mean(valid_r2s):.4f}, Std={np.std(valid_r2s):.4f}")
            print(f"Dataset {i} - Valid Custom Score: Mean={np.mean(valid_custom_scores):.4f}, Std={np.std(valid_custom_scores):.4f}")
            print('\n')
            # 对测试集进行预测
            predictions = self.svc_predict(i, data = X_test)
            predictions = np.clip(predictions, 0, 1)

            task_results = pd.DataFrame({
                'id': self.test_data[i].iloc[:, 0],
                'label': predictions
            })

            all_results = pd.concat([all_results, task_results], ignore_index=True)

        output_csv = 'result.csv'
        all_results.to_csv(output_csv, index=False)

    def custom_scorer(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return 0.5 * mae + 0.5 * rmse

    def _plot_comparison(self, train_maes, valid_maes, train_rmses, valid_rmses, train_r2s, valid_r2s, train_custom_scores, valid_custom_scores):
        plt.figure(figsize=(18, 12))

        # MAE 对比
        plt.subplot(2, 2, 1)
        plt.plot(train_maes, label='Train MAE', marker='o')
        plt.plot(valid_maes, label='Validation MAE', marker='o')
        plt.xlabel('Fold Number')
        plt.ylabel('MAE')
        plt.title('Train vs Validation MAE')
        plt.legend()

        # RMSE 对比
        plt.subplot(2, 2, 2)
        plt.plot(train_rmses, label='Train RMSE', marker='o', color='orange')
        plt.plot(valid_rmses, label='Validation RMSE', marker='o', color='red')
        plt.xlabel('Fold Number')
        plt.ylabel('RMSE')
        plt.title('Train vs Validation RMSE')
        plt.legend()

        # R² 对比
        plt.subplot(2, 2, 3)
        plt.plot(train_r2s, label='Train R² Score', marker='o', color='green')
        plt.plot(valid_r2s, label='Validation R² Score', marker='o', color='blue')
        plt.xlabel('Fold Number')
        plt.ylabel('R² Score')
        plt.title('Train vs Validation R² Score')
        plt.legend()

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


model = Model()
