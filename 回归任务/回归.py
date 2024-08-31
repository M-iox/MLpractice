from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier,StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier

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

        self.train_features=self.train_data
        self.test_features=self.test_data
        self.labels = {1: 'label1', 2: 'label2', 3: 'label3'}

        self.process_data()

        # for i in range(1, 4):
        #     print(f"self.train_features[i].isna():{self.train_features[i].isna().sum()}")
        #     print(f"self.test_features[i].isna():{self.test_features[i].isna().sum()}")
        #
        for i in range(1, 4):
            print(f"self.train_data[i].shape:{self.train_data[i].shape}")
            print(f"self.test_data[i].shape:{self.test_data[i].shape}")
        # for i in range(1,4):
        #     print(f"self.train_data[i]:{self.train_data[i]}")
        #
        #     print(f"self.test_data[i]:{self.test_data[i]}")

        # for i in range(1,4):
        #
        #
        #     print(self.train_features[i])
        #     print(self.test_features[i])
        #     print(self.labels)

            #
            # self.tuned_para()
            # self.train_test(self.train_features,'train')
            # self.train_test(self.test_features, 'test')

    # def _print_results(self, accuracies, f1_scores, custom_scores):
    #     print(f'Custom scores (0.5*Acc + 0.5*F1): {custom_scores}')
    #     print(f'Mean custom score: {np.mean(custom_scores):.4f}')
    #     print(f'Standard deviation: {np.std(custom_scores):.4f}')
    #
    # def _plot_results(self, accuracies, f1_scores, custom_scores):
    #     plt.figure(figsize=(15, 6))
    #
    #     # 自定义评分
    #     plt.bar(range(1, len(custom_scores) + 1), custom_scores, color='purple', alpha=0.7)
    #     plt.xlabel('Fold Number')
    #     plt.ylabel('Custom Score')
    #     plt.title('Cross-Validation Custom Score for Each Fold')
    #     plt.ylim(0.8, 1.0)
    #     for i in range(len(custom_scores)):
    #         plt.text(i + 1, custom_scores[i], f'{custom_scores[i]:.4f}', ha='center', va='bottom')
    #
    #     plt.tight_layout()
    #     plt.show()
    #
    # def custom_scorer(self, y_true, y_pred):
    #     acc = accuracy_score(y_true, y_pred)
    #     f1 = f1_score(y_true, y_pred, average='weighted')
    #     return 0.5 * acc + 0.5 * f1
    def process_label(self):
        self.labels[1] = self.train_data[1]['Deceased'] #No存活 ，Yes死亡
        self.labels[2] = self.train_data[2]['outcome'] #1表示存活, 2表示死亡
        self.labels[3] = self.train_data[3]['Death (1 Yes 2 No)'] #1死亡，2存活
        # label_encoder = LabelEncoder()
        # self.labels = label_encoder.fit_transform(self.labels)


        # for i in range(1,4):
        #     print(self.labels[i])

    def process_data(self):
        # 初始化独热编码所需的列
        self.encoder_columns = None

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





        # #标准化或正则化
        # scaler = StandardScaler()
        # data=scaler.fit_transform(data)
        #
        # #使用PCA降维
        # pca = PCA(n_components=0.95)  # 保留95%的方差
        # data = pca.fit_transform(data)
        # # Mean custom score: 0.9596
        # # Standard deviation: 0.0249
        #
        # # 将 numpy 数组转换回 DataFrame
        # data = pd.DataFrame(data)
        #
        # return data
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
        self.svc1 = SVC(kernel='rbf', C=1, gamma='scale', probability=True, random_state=42)  # 使用RBF核函数
        self.svc2 = SVC(kernel='linear', C=0.1, probability=True, random_state=42)  # 使用线性核函数
        self.svc3 = SVC(kernel='poly', degree=3, C=0.5, gamma='scale', probability=True, random_state=42)  # 使用多项式核函数
        self.rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)
        self.gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
        self.lr = LogisticRegression(penalty='l2', C=0.5, random_state=42, max_iter=1000)
        self.et = ExtraTreesClassifier(n_estimators=100, random_state=42)

    def default_para(self):
        self.ensemble_model()
        # self._model = StackingClassifier(
        #     estimators=[
        #         ('svc1', self.svc1),
        #         # ('rf', self.rf),
        #         ('svc2', self.svc2),
        #         ('svc3', self.svc3),
        #         ('gb', self.gb)
        #     ],
        #     final_estimator=GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42),
        #     cv=5,  # 使用5折交叉验证
        #     n_jobs=-1,  # 使用所有CPU核进行并行计算
        #     passthrough=False # 将原始特征传递给元学习器
        # )
        self._model = self.svc1

    def tuned_para(self):
        param_grid = {
            'n_estimators': [100, 150, 200, 500, 700, 900],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [4, 6, 8, 12, 14, 16],
            'criterion': ['gini', 'entropy'],
            'n_jobs': [-1, 1, None]
        }
        grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv= 5)
        grid_search.fit(self.train_features,self.labels)
        print("Best parameters found: ", grid_search.best_params_)
        self._model = RandomForestClassifier(** grid_search.best_params_)

    def train_test(self, data, mode):
        if mode == 'train':
            kf = KFold(n_splits=10, shuffle=True, random_state=42)
            accuracies, f1_scores, custom_scores = [], [], []
            smote = SMOTE(random_state=42)

            for train_index, valid_index in kf.split(data):
                X_train_fold, X_valid_fold = data.iloc[train_index], data.iloc[valid_index]
                y_train_fold, y_valid_fold = self.labels[train_index], self.labels[valid_index]

                # 应用SMOTE来平衡训练数据
                X_train_fold, y_train_fold = smote.fit_resample(X_train_fold, y_train_fold)

                # 输出SMOTE后的样本分布
                print(f"Fold {kf.get_n_splits()}: Resampled class distribution: {Counter(y_train_fold)}")

                self._model.fit(X_train_fold, y_train_fold)
                y_pred_fold = self._model.predict(X_valid_fold)
                acc = accuracy_score(y_valid_fold, y_pred_fold)
                f1 = f1_score(y_valid_fold, y_pred_fold, average='weighted')
                custom = self.custom_scorer(y_valid_fold, y_pred_fold)

                accuracies.append(acc)
                f1_scores.append(f1)
                custom_scores.append(custom)

            self._print_results(accuracies, f1_scores, custom_scores)
            self._plot_results(accuracies, f1_scores, custom_scores)

        if mode == 'test':
            predictions = self._model.predict(data)
            results = pd.DataFrame({'id': self.test_data.iloc[:, 0], 'Group': predictions})
            output_csv = 'result.csv'
            results.to_csv(output_csv, index=False)

model = Model()
