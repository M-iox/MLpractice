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



class model:
    def __init__(self):
        # 加载训练集数据
        self.file_path = './train3.xlsx'
        self.train_data = pd.read_excel(self.file_path)
        self.X_train = self.train_data.drop(columns=['Number', 'Death (1 Yes 2 No)'])
        self.y_train = self.train_data['Death (1 Yes 2 No)'].apply(lambda x: 1 if x == 1 else 0)

        # 加载测试数据
        self.test_file_path = './test3.csv'
        self.test_data_processed = pd.read_csv(self.test_file_path, encoding='ISO-8859-1')
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

        print(self.X_train.shape)
        # 特征选择
        self.X_train, self.test_data_processed = self.select_important_features()

        print(self.X_train.shape)
        # pca和标准化
        self.X_train = self.standardize_and_reduce(self.X_train, fit=True)
        self.test_data_processed = self.standardize_and_reduce(self.test_data_processed, fit=False)

        # 训练和预测
        self.train_test()

    def data_process(self, X_train):
        # 删除处理
        empty = ['D1 blood routine upon admission', 'D7 blood routine', 'Liver enzyme D1', 'Coagulation index D1',
                 'Inflammatory indicator D1', 'Blood gas analysis', 'Virus nucleic acid test CT value',
                 'Pulmonary CT condition']
        X_train = X_train.drop(columns=empty)

        # 标签处理或独热的列
        str_columns = ['Age ranges[years]', 'Specimen collection location', 'Specimen collection location',
                       'Pathogenic culture results',
                       'Basic diseases (1 hypertension, 2 diabetes, 3 cardiovascular diseases, 4 cerebrovascular diseases, 5 COPD, 6 immunodeficiency (such as AIDS, hormone, immunosuppressant use history), 7 malignant tumor, 8 other 9 chronic kidney disease']

        # 大部分为int的列
        all_columns = X_train.columns.tolist()
        int_columns = [col for col in all_columns if col not in str_columns]

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
        X_train = X_train[str_columns].replace({'[null]': '0', '/': '0', '-': '0'})

        # 处理分类变量和缺失值
        X_train = pd.get_dummies(X_train, columns=str_columns, dummy_na=True, dtype=int, drop_first=True).fillna(0)

        return X_train

    def select_important_features(self):
        # 使用随机森林进行特征选择
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.X_train, self.y_train)

        # 使用SelectFromModel选择最重要的特征
        selector = SelectFromModel(rf, threshold='mean', prefit=True)
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
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],  # 核函数类型
            'degree': [3, 4, 5]  # 仅对多项式核有效
        }

        svm = SVC(probability=True, random_state=42)

        # 使用 GridSearchCV 进行超参数调优
        grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=1, verbose=2)
        grid_search.fit(self.X_train, self.y_train)

        print(f"Best parameters found: {grid_search.best_params_}")
        self._model = grid_search.best_estimator_

    def default_para(self):
        self._model = SVC(probability=True, random_state=42, C=1, degree= 3, gamma= 0.01, kernel= 'rbf')
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

    def train_test(self):
        # 将训练数据分割为训练集和验证集
        X_train_split, X_valid, y_train_split, y_valid = train_test_split(self.X_train, self.y_train, test_size=0.2,
                                                                          random_state=42)

        self._model = SVC(probability=True, random_state=42, C=1, degree= 3, gamma= 0.01, kernel= 'rbf')
        self._model.fit(X_train_split, y_train_split)
        self.calibrated_svm = CalibratedClassifierCV(estimator=self._model, method='sigmoid', cv='prefit')
        self.calibrated_svm.fit(X_train_split, y_train_split)

        # 验证模型
        y_valid_pred = self.calibrated_svm.predict_proba(X_valid)[:, 1]
        validation_error = mean_absolute_error(y_valid, y_valid_pred)
        validation_error_rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred))

        print("Validation Error (MAE):", validation_error)
        print("Validation Error (RMSE):", validation_error_rmse)

        # 在测试集上进行预测
        y_test_pred = self.calibrated_svm.predict_proba(self.test_data_processed)[:, 1]

        # 准备预测结果
        predictions = pd.DataFrame({
            'id': self.id_number,
            'label': y_test_pred
        })

        # 将预测结果保存为CSV文件
        output_file_path = './result3.csv.csv'
        predictions.to_csv(output_file_path, index=False)


Model = model()
