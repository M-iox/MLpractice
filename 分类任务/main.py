from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier,StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class Model:
    def __init__(self):
        self.train_data = pd.read_csv('train.csv')
        self.test_data = pd.read_csv('test.csv')
        self.train_features,self.test_features = self.process_data()

        self.default_para()

        self.train_test(self.train_features,'train')
        self.train_test(self.test_features, 'test')

    def _print_results(self, accuracies, f1_scores, custom_scores):
        print(f'Custom scores (0.5*Acc + 0.5*F1): {custom_scores}')
        print(f'Mean custom score: {np.mean(custom_scores):.4f}')
        print(f'Standard deviation: {np.std(custom_scores):.4f}')
    def custom_scorer(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        return 0.5 * acc + 0.5 * f1
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

    def process_data(self):
        self.train_features = self.train_data.copy()
        self.test_features = self.test_data.copy()

        # 去掉列名的空格
        self.train_features.rename(columns=lambda x: x.strip(), inplace=True)
        self.test_features.rename(columns=lambda x: x.strip(), inplace=True)

        # 将 Troponin 列中的非数值数据转换为数值类型，'Negative' 转换为 0
        self.train_features['Troponin'] = self.train_features['Troponin'].apply(
            lambda x: 0 if x == 'Negative' else float(x) if isinstance(x, str) and x.replace('.', '',
                                                                                             1).isdigit() else x)
        self.test_features['Troponin'] = self.train_features['Troponin'].apply(
            lambda x: 0 if x == 'Negative' else float(x) if isinstance(x, str) and x.replace('.', '',

                                                                                             1).isdigit() else x)

        # 数据预处理：提取特征和标签
        self.labels = self.train_features['Group'].apply(lambda x: 1 if x != 'Control' else 0)  # 转换标签为0和1
        self.train_features = self.train_features.drop(columns=['Patient Code', 'Group'])  # 去掉患者编码和标签

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
            train_accuracies, valid_accuracies = [], []
            train_f1_scores, valid_f1_scores = [], []
            train_custom_scores, valid_custom_scores = [], []

            # 保存所有验证集的真实标签和预测标签
            all_y_valid_true = []
            all_y_valid_pred = []

            for train_index, valid_index in kf.split(data):
                X_train_fold, X_valid_fold = data.iloc[train_index], data.iloc[valid_index]
                y_train_fold, y_valid_fold = self.labels[train_index], self.labels[valid_index]

                self._model.fit(X_train_fold, y_train_fold)

                # 预测验证集
                y_valid_pred_fold = self._model.predict(X_valid_fold)

                # 保存验证集真实值和预测值
                all_y_valid_true.extend(y_valid_fold)
                all_y_valid_pred.extend(y_valid_pred_fold)

                # 计算训练集和验证集的准确率和 F1 分数
                train_acc = accuracy_score(y_train_fold, self._model.predict(X_train_fold))
                valid_acc = accuracy_score(y_valid_fold, y_valid_pred_fold)
                train_f1 = f1_score(y_train_fold, self._model.predict(X_train_fold), average='weighted')
                valid_f1 = f1_score(y_valid_fold, y_valid_pred_fold, average='weighted')

                # 计算自定义评分
                train_custom = self.custom_scorer(y_train_fold, self._model.predict(X_train_fold))
                valid_custom = self.custom_scorer(y_valid_fold, y_valid_pred_fold)

                train_accuracies.append(train_acc)
                valid_accuracies.append(valid_acc)
                train_f1_scores.append(train_f1)
                valid_f1_scores.append(valid_f1)
                train_custom_scores.append(train_custom)
                valid_custom_scores.append(valid_custom)

            # 打印并绘制训练集和验证集的结果
            self._plot_comparison(train_accuracies, valid_accuracies, train_f1_scores, valid_f1_scores,
                                  train_custom_scores, valid_custom_scores)
            self._print_results(valid_accuracies, valid_f1_scores, valid_custom_scores)

            # 计算并绘制所有折叠验证集的混淆矩阵
            cm = confusion_matrix(all_y_valid_true, all_y_valid_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Blues')
            plt.title('Final Confusion Matrix (All Folds)')
            plt.show()

        if mode == 'test':
            predictions = self._model.predict(data)
            print(f'Prediction distribution: {Counter(predictions)}')
            results = pd.DataFrame({'id': self.test_data.iloc[:, 0], 'Group': predictions})
            output_csv = 'result.csv'
            results.to_csv(output_csv, index=False)


model = Model()
