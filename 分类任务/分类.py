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
        self.train_data = pd.read_csv('train.csv')
        self.test_data = pd.read_csv('test.csv')
        self.train_features = self.process_data(self.train_data, mode='train')
        self.test_features  = self.process_data(self.test_data, mode='test')

        print(self.train_features)
        print(self.test_features)
        print(self.labels)

        self.default_para()
        self.train_test(self.train_features,'train')
        self.train_test(self.test_features, 'test')


    def _print_results(self, accuracies, f1_scores, custom_scores):
        print(f'Custom scores (0.5*Acc + 0.5*F1): {custom_scores}')
        print(f'Mean custom score: {np.mean(custom_scores):.4f}')
        print(f'Standard deviation: {np.std(custom_scores):.4f}')

    def _plot_results(self, accuracies, f1_scores, custom_scores):
        plt.figure(figsize=(15, 6))

        # 自定义评分
        plt.bar(range(1, len(custom_scores) + 1), custom_scores, color='purple', alpha=0.7)
        plt.xlabel('Fold Number')
        plt.ylabel('Custom Score')
        plt.title('Cross-Validation Custom Score for Each Fold')
        plt.ylim(0.8, 1.0)
        for i in range(len(custom_scores)):
            plt.text(i + 1, custom_scores[i], f'{custom_scores[i]:.4f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    def custom_scorer(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        return 0.5 * acc + 0.5 * f1


    def process_data(self, data, mode):
        data.columns = data.columns.str.strip()  # 去掉列名的空格
        data = data.drop(columns=['IL-6'], errors='ignore')  # 删除特定列，忽略不存在的列
        # 处理数据
        if mode == 'train':
            # 训练数据处理逻辑
            # 处理标签
            self.labels = data['Group']
            label_encoder = LabelEncoder()
            self.labels = label_encoder.fit_transform(self.labels)
            self.labels = np.where(self.labels == 0, 1, 0)

            data = data.iloc[:, 2:]

        elif mode == 'test':
            # 测试数据处理逻辑
            data = data.iloc[:, 1:]

        # 处理缺失值
        numeric_features = data.dtypes[data.dtypes != 'object'].index
        categorical_features = data.select_dtypes(include=['object']).columns

        numeric_imputer = SimpleImputer(strategy='median')
        categorical_imputer = SimpleImputer(strategy='most_frequent')

        data[numeric_features] = numeric_imputer.fit_transform(data[numeric_features])
        data[categorical_features] = categorical_imputer.fit_transform(data[categorical_features])

        # One-Hot编码处理离散值
        data = pd.get_dummies(data, dummy_na=True, dtype=int)

        #标准化或正则化
        scaler = StandardScaler()
        data=scaler.fit_transform(data)


        #使用PCA降维
        pca = PCA(n_components=0.95)  # 保留95%的方差
        data = pca.fit_transform(data)
        # Mean custom score: 0.9596
        # Standard deviation: 0.0249

        # 将 numpy 数组转换回 DataFrame
        data = pd.DataFrame(data)

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
        self._model = StackingClassifier(
            estimators=[
                ('svc1', self.svc1),
                ('rf', self.rf),
                ('svc2', self.svc2),
                ('svc3', self.svc3),
                ('gb', self.gb)
            ],
            final_estimator=self.lr,
            cv=10,  # 使用5折交叉验证
            n_jobs=-1,  # 使用所有CPU核进行并行计算
            passthrough=False # 将原始特征传递给元学习器
        )
        # self._model = self.svc1

    def tuned_para(self):
        param_grid = {
            'n_estimators': [30, 50, 100],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [4, 6, 8, 12, 14, 16],
            'criterion': ['gini', 'entropy']
        }
        grid_search = GridSearchCV(estimator=RandomForestClassifier(n_jobs=-1), param_grid=param_grid, cv=5)
        grid_search.fit(self.train_features, self.labels)
        print("Best parameters found: ", grid_search.best_params_)
        self._model = RandomForestClassifier(**grid_search.best_params_, n_jobs=-1)

    def train_test(self, data, mode):
        if mode == 'train':
            kf = KFold(n_splits=20, shuffle=True, random_state=42)
            train_accuracies, valid_accuracies = [], []
            train_f1_scores, valid_f1_scores = [], []
            train_custom_scores, valid_custom_scores = [], []

            smote = SMOTE(random_state=42)

            for train_index, valid_index in kf.split(data):
                X_train_fold, X_valid_fold = data.iloc[train_index], data.iloc[valid_index]
                y_train_fold, y_valid_fold = self.labels[train_index], self.labels[valid_index]

                # 应用SMOTE来平衡训练数据
                X_train_fold, y_train_fold = smote.fit_resample(X_train_fold, y_train_fold)

                # 输出SMOTE后的样本分布
                print(f"Fold {kf.get_n_splits()}: Resampled class distribution: {Counter(y_train_fold)}")

                self._model.fit(X_train_fold, y_train_fold)

                # 预测训练集和验证集
                y_train_pred_fold = self._model.predict(X_train_fold)
                y_valid_pred_fold = self._model.predict(X_valid_fold)

                # 计算训练集和验证集的准确率和 F1 分数
                train_acc = accuracy_score(y_train_fold, y_train_pred_fold)
                valid_acc = accuracy_score(y_valid_fold, y_valid_pred_fold)
                train_f1 = f1_score(y_train_fold, y_train_pred_fold, average='weighted')
                valid_f1 = f1_score(y_valid_fold, y_valid_pred_fold, average='weighted')

                # 计算自定义评分
                train_custom = self.custom_scorer(y_train_fold, y_train_pred_fold)
                valid_custom = self.custom_scorer(y_valid_fold, y_valid_pred_fold)

                train_accuracies.append(train_acc)
                valid_accuracies.append(valid_acc)
                train_f1_scores.append(train_f1)
                valid_f1_scores.append(valid_f1)
                train_custom_scores.append(train_custom)
                valid_custom_scores.append(valid_custom)

            # 打印并绘制训练集和验证集的结果
            self._plot_comparison(train_accuracies, valid_accuracies, train_f1_scores, valid_f1_scores, train_custom_scores, valid_custom_scores)
            self._print_results(valid_accuracies, valid_f1_scores, valid_custom_scores)

        if mode == 'test':
            predictions = self._model.predict(data)
            results = pd.DataFrame({'id': self.test_data.iloc[:, 0], 'Group': predictions})
            output_csv = 'result.csv'
            results.to_csv(output_csv, index=False)
    def _plot_comparison(self, train_acc, valid_acc, train_f1, valid_f1, train_custom, valid_custom):
        plt.figure(figsize=(20, 6))

        # 准确率对比
        plt.subplot(1, 3, 1)
        plt.plot(range(1, len(train_acc) + 1), train_acc, label='Train Accuracy', marker='o')
        plt.plot(range(1, len(valid_acc) + 1), valid_acc, label='Validation Accuracy', marker='o')
        plt.xlabel('Fold Number')
        plt.ylabel('Accuracy')
        plt.title('Train vs Validation Accuracy')
        plt.legend()

        # F1 分数对比
        plt.subplot(1, 3, 2)
        plt.plot(range(1, len(train_f1) + 1), train_f1, label='Train F1 Score', marker='o')
        plt.plot(range(1, len(valid_f1) + 1), valid_f1, label='Validation F1 Score', marker='o')
        plt.xlabel('Fold Number')
        plt.ylabel('F1 Score')
        plt.title('Train vs Validation F1 Score')
        plt.legend()

        # 自定义评分对比
        plt.subplot(1, 3, 3)
        plt.plot(range(1, len(train_custom) + 1), train_custom, label='Train Custom Score', marker='o')
        plt.plot(range(1, len(valid_custom) + 1), valid_custom, label='Validation Custom Score', marker='o')
        plt.xlabel('Fold Number')
        plt.ylabel('Custom Score')
        plt.title('Train vs Validation Custom Score')
        plt.legend()

        plt.tight_layout()
        plt.show()


model = Model()
