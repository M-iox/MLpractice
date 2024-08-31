from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
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
        # train2 和 train3 是 Excel 文件
        self.train_data[2] = pd.read_excel('train2.xlsx', sheet_name='Sheet1')
        self.train_data[3] = pd.read_excel('train3.xlsx', sheet_name='Sheet1')

        self.train_features = {i: self.train_data[i].copy() for i in range(1, 4)}
        self.test_features = {i: self.test_data[i].copy() for i in range(1, 4)}

        self.labels = {1: 'label1', 2: 'label2', 3: 'label3'}

        self.process_data()

        for i in range(1, 4):
            print(f"self.train_data[i].shape:{self.train_data[i].shape}")
            print(f"self.test_data[i].shape:{self.test_data[i].shape}")

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

        # 处理缺失值和独热编码
        for i in range(1, 4):
            self.train_features[i] = self.handle_missing_and_encode(self.train_features[i], fit=True)
            self.test_features[i] = self.handle_missing_and_encode(self.test_features[i], fit=False)
            # 标准化和PCA降维
            self.train_features[i] = self.standardize_and_reduce(self.train_features[i], fit=True)
            self.test_features[i] = self.standardize_and_reduce(self.test_features[i], fit=False)

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
        self.rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)
        self.gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
        self.lr = LinearRegression()
        self.et = ExtraTreesRegressor(n_estimators=100, random_state=42)

    def default_para(self):
        self.ensemble_model()
        # # 修改为StackingRegressor或其他适合回归的集成方法
        # self._model = StackingRegressor(
        #     estimators=[
        #         ('svr', self.svr),
        #         ('rf', self.rf),
        #         ('gb', self.gb)
        #     ],
        #     final_estimator=GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42),
        #     cv=5,
        #     n_jobs=-1,
        #     passthrough=False
        # )
        self._model = self.svr
    def tuned_para(self):
        param_grid = {
            'n_estimators': [100, 150, 200, 500, 700, 900],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [4, 6, 8, 12, 14, 16]
        }
        grid_search = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid, cv=5)
        grid_search.fit(self.train_features[1], self.labels[1])  # 确保使用正确的特征和标签
        print("Best parameters found: ", grid_search.best_params_)
        self._model = RandomForestRegressor(**grid_search.best_params_)

    def train_and_predict(self):
        # 创建一个空的DataFrame来存储所有预测结果
        all_results = pd.DataFrame()

        # 遍历每个任务进行训练和预测
        for i in range(1, 4):
            # 获取训练和测试数据
            X_train = self.train_features[i]
            y_train = self.labels[i]
            X_test = self.test_features[i]

            # 进行交叉验证训练模型
            kf = KFold(n_splits=10, shuffle=True, random_state=42)
            mse_scores, mae_scores, r2_scores = [], [], []

            for train_index, valid_index in kf.split(X_train):
                X_train_fold, X_valid_fold = X_train.iloc[train_index], X_train.iloc[valid_index]
                y_train_fold, y_valid_fold = y_train.iloc[train_index], y_train.iloc[valid_index]

                # 训练模型
                self._model.fit(X_train_fold, y_train_fold)

                # 验证模型
                y_pred_fold = self._model.predict(X_valid_fold)
                mse = mean_squared_error(y_valid_fold, y_pred_fold)
                mae = mean_absolute_error(y_valid_fold, y_pred_fold)
                r2 = r2_score(y_valid_fold, y_pred_fold)

                # 保存每折的评分
                mse_scores.append(mse)
                mae_scores.append(mae)
                r2_scores.append(r2)

            # 打印和绘制结果
            self._print_results(mae_scores, mse_scores, r2_scores)
            self._plot_results(mae_scores, mse_scores, r2_scores)

            # 使用训练后的模型预测测试集
            predictions = self._model.predict(X_test)

            # 将预测结果限制在0到1之间
            predictions = np.clip(predictions, 0, 1)

            # 将当前任务的结果存储在DataFrame中
            task_results = pd.DataFrame({
                'id': self.test_data[i].iloc[:, 0],  # 假设id是第一列
                'label': predictions  # 使用'label'表示死亡率
            })

            # 将当前任务的结果添加到最终结果的DataFrame中
            all_results = pd.concat([all_results, task_results], ignore_index=True)

        # 将所有结果保存到CSV文件
        output_csv = 'result.csv'
        all_results.to_csv(output_csv, index=False)



    def _print_results(self, maes, rmses, custom_scores):
        print(f'Custom scores (0.5*MAE + 0.5*RMSE): {custom_scores}')
        print(f'Mean custom score: {np.mean(custom_scores):.4f}')
        print(f'Standard deviation: {np.std(custom_scores):.4f}')

    def _plot_results(self, maes, rmses, custom_scores):
        plt.figure(figsize=(15, 6))
        # 自定义评分
        plt.bar(range(1, len(custom_scores) + 1), custom_scores, color='purple', alpha=0.7)
        plt.xlabel('Fold Number')
        plt.ylabel('Custom Score')
        plt.title('Cross-Validation Custom Score for Each Fold')
        for i in range(len(custom_scores)):
            plt.text(i + 1, custom_scores[i], f'{custom_scores[i]:.4f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    def custom_scorer(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return 0.5 * mae + 0.5 * rmse


model = Model()
