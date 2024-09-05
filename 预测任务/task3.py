from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel

class Model:
    def __init__(self):
        self.train_data = pd.read_excel('train3.xlsx')
        self.test_data = pd.read_csv('test3.csv', encoding='latin1')
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
        plt.plot(train_custom_scores, label='Train Custom Score', marker='o', color='purple')
        plt.plot(valid_custom_scores, label='Validation Custom Score', marker='o', color='pink')
        plt.xlabel('Fold Number')
        plt.ylabel('Custom Score')
        plt.title('Train vs Validation Custom Score')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def process_data(self):
        # 复制数据
        self.train_features = self.train_data.copy()
        self.test_features = self.test_data.copy()

        # 数据预处理：提取特征和标签
        self.labels = self.train_features['Death (1 Yes 2 No)'].apply(lambda x: 1 if x == 1 else 0)  # 转换标签为0和1
        self.train_features = self.train_features.drop(columns=['Number', 'Death (1 Yes 2 No)'])  # 去掉患者编码和标签
        self.test_features = self.test_features.drop(columns=['Patient Code'])  # 去掉测试集中的患者编码

        # 统一测试集中的特征名称
        self.test_features = self.test_features.rename(columns={
            'Patient Code': 'Number',
            'Disease onset D1 highest body temperature[¡æ]': 'Disease onset D1 highest body temperature[℃]',
            'creatinine[¦Ìmol/L]': 'creatinine[μmol/L]',
            'total bilirubin[¦Ìmol/L]': 'total bilirubin[μmol/L]'
        })

        # 获取所有 object 类型的列
        train_object_columns = self.train_features.select_dtypes(include=['object']).columns
        test_object_columns = self.test_features.select_dtypes(include=['object']).columns

        # 转换 object 类型的列为数值型，记录无法转换的列
        train_conversion_issues = []
        test_conversion_issues = []

        for col in train_object_columns:
            try:
                self.train_features[col] = pd.to_numeric(self.train_features[col], errors='raise')
            except ValueError:
                train_conversion_issues.append(col)

        for col in test_object_columns:
            try:
                self.test_features[col] = pd.to_numeric(self.test_features[col], errors='raise')
            except ValueError:
                test_conversion_issues.append(col)

        # 对无法转换的列进行处理，使用 LabelEncoder 对分类特征编码
        label_encoders = {}
        for col in train_conversion_issues:
            if self.train_features[col].dtype == 'object':
                le = LabelEncoder()
                self.train_features[col] = le.fit_transform(self.train_features[col].astype(str))
                self.test_features[col] = self.test_features[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1)
                label_encoders[col] = le

        # 再次尝试将这些列转换为数值型
        for col in train_conversion_issues:
            self.train_features[col] = pd.to_numeric(self.train_features[col], errors='coerce')
            self.test_features[col] = pd.to_numeric(self.test_features[col], errors='coerce')

        # 针对测试集中剩余的 object 类型列 `NRL[%]` 进行处理
        if 'NRL[%]' in self.test_features.columns:
            self.test_features['NRL[%]'] = pd.to_numeric(self.test_features['NRL[%]'], errors='coerce')

        # 检查是否有全为 NaN 的列
        nan_columns_train = self.train_features.columns[self.train_features.isna().all()].tolist()
        nan_columns_test = self.test_features.columns[self.test_features.isna().all()].tolist()

        # 如果有完全为 NaN 的列，手动处理这些列（如删除或填充）
        if nan_columns_train or nan_columns_test:
            print("Columns with all NaN in train:", nan_columns_train)
            print("Columns with all NaN in test:", nan_columns_test)
            self.train_features.drop(columns=nan_columns_train, inplace=True)
            self.test_features.drop(columns=nan_columns_test, inplace=True)

        # 处理缺失值
        common_columns = self.train_features.columns.intersection(self.test_features.columns)
        self.train_features = self.train_features[common_columns]
        self.test_features = self.test_features[common_columns]

        imputer = SimpleImputer(strategy='mean')
        self.train_features = pd.DataFrame(imputer.fit_transform(self.train_features),
                                           columns=common_columns)
        self.test_features = pd.DataFrame(imputer.transform(self.test_features),
                                          columns=common_columns)


        return self.train_features, self.test_features

    def default_para(self):
        # 使用随机森林进行特征选择
        feature_selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
        # 使用选定的PCA主成分数目
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),  # 使用均值填补缺失值
            ('scaler', StandardScaler()),  # 数据标准化
            ('feature_selector', feature_selector),  # 加入特征选择步骤
            # ('pca', PCA(n_components=self.n_components)),  # 使用推荐的主成分数目
            ('classifier', LogisticRegression(random_state=42, penalty='l2', solver='liblinear', C=1))
        ])
        self._model = pipeline

    def determine_pca_components(self):
        # 标准化数据
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.train_features)

        # 使用PCA进行降维
        pca = PCA().fit(scaled_features)

        # 累积解释方差比例
        explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

        # 绘制累积解释方差曲线
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='--')
        plt.title('Cumulative Explained Variance by Number of Principal Components')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.grid()
        plt.show()

        # 打印解释方差达到90%和95%时对应的主成分数目
        n_components_90 = np.argmax(explained_variance_ratio >= 0.90) + 1
        n_components_95 = np.argmax(explained_variance_ratio >= 0.95) + 1

        print(f'Number of components for 90% explained variance: {n_components_90}')
        print(f'Number of components for 95% explained variance: {n_components_95}')

        # 返回推荐的主成分数目
        self.n_components = n_components_95
    def tuned_para(self):
        self.default_para()
        pass

    def train_test(self, data, mode):
        if mode == 'train':
            kf = KFold(n_splits=10, shuffle=True, random_state=42)
            loo = LeaveOneOut()
            train_maes, valid_maes = [], []
            train_rmses, valid_rmses = [], []
            train_custom_scores, valid_custom_scores = [], []

            for train_index, valid_index in kf.split(data):
                X_train_fold, X_valid_fold = data.iloc[train_index], data.iloc[valid_index]
                y_train_fold, y_valid_fold = self.labels[train_index], self.labels[valid_index]

                # self._model.fit(X_train_fold, y_train_fold)
                # 应用 SMOTE 仅在训练数据上
                smote = SMOTE(random_state=42)
                X_train_fold_resampled, y_train_fold_resampled = smote.fit_resample(X_train_fold, y_train_fold)

                self._model.fit(X_train_fold_resampled, y_train_fold_resampled)

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
            self._model.fit(self.train_features, self.labels)
            calibrated_model = CalibratedClassifierCV(self._model, method='sigmoid', cv='prefit')
            calibrated_model.fit(self.train_features, self.labels)
            predictions_proba = calibrated_model.predict_proba(data)[:, 1]

            predictions = self._model.predict(data)
            results = pd.DataFrame({'id': self.test_data.iloc[:, 0], 'label': predictions_proba})
            output_csv = 'result3.csv'
            results.to_csv(output_csv, index=False)

model = Model()
