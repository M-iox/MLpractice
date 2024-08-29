from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class RandomForestModel:
    def __init__(self, data_path, sheet_name=0):
        self.data_path = data_path
        self.sheet_name = sheet_name
        self.train_data = pd.read_excel(self.data_path, sheet_name=self.sheet_name)
        self.train_data.columns = self.train_data.columns.str.strip()
        self.features = self.train_data.iloc[:, 2:-1]
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.array(['Control', 'COVID-19'])
        self.labels = self.label_encoder.transform(self.train_data['Group'].values.ravel())
        self.process_data()

    def process_data(self):
        # 填充数值型和分类特征的NA值
        numeric_features = self.features.dtypes[self.features.dtypes != 'object'].index
        numeric_imputer = SimpleImputer(strategy='median')
        self.features[numeric_features] = numeric_imputer.fit_transform(self.features[numeric_features])

        categorical_features = self.features.columns.difference(numeric_features)
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.features[categorical_features] = categorical_imputer.fit_transform(self.features[categorical_features])

        # One-Hot编码处理离散值
        self.features = pd.get_dummies(self.features, dummy_na=True, dtype=int)

    def custom_scorer(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        return 0.5 * acc + 0.5 * f1

    def tune_hyperparameters(self):
        # 超参数调优
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'max_features': ['auto', 'sqrt', 'log2']
        }

        grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.features, self.labels)
        print("Best parameters found: ", grid_search.best_params_)
        self.rf_model = RandomForestClassifier(**grid_search.best_params_, random_state=42)

    def default_parameters(self):
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    def cross_validate(self):
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        accuracies, f1_scores, custom_scores = [], [], []

        for train_index, test_index in kf.split(self.features):
            X_train_fold, X_test_fold = self.features.iloc[train_index], self.features.iloc[test_index]
            y_train_fold, y_test_fold = self.labels[train_index], self.labels[test_index]

            self.rf_model.fit(X_train_fold, y_train_fold)
            y_pred_fold = self.rf_model.predict(X_test_fold)

            acc = accuracy_score(y_test_fold, y_pred_fold)
            f1 = f1_score(y_test_fold, y_pred_fold, average='weighted')
            custom = self.custom_scorer(y_test_fold, y_pred_fold)

            accuracies.append(acc)
            f1_scores.append(f1)
            custom_scores.append(custom)

        self._print_results(accuracies, f1_scores, custom_scores)
        self._plot_results(accuracies, f1_scores, custom_scores)


    def _print_results(self, accuracies, f1_scores, custom_scores):
        print(f'Accuracy scores: {accuracies}')
        print(f'Mean accuracy: {np.mean(accuracies):.4f}')
        print(f'Standard deviation: {np.std(accuracies):.4f}')

        print(f'F1 scores: {f1_scores}')
        print(f'Mean F1 score: {np.mean(f1_scores):.4f}')
        print(f'Standard deviation: {np.std(f1_scores):.4f}')

        print(f'Custom scores (0.5*Acc + 0.5*F1): {custom_scores}')
        print(f'Mean custom score: {np.mean(custom_scores):.4f}')
        print(f'Standard deviation: {np.std(custom_scores):.4f}')

    def _plot_results(self, accuracies, f1_scores, custom_scores):
        plt.figure(figsize=(15, 6))

        # 准确率
        plt.subplot(1, 3, 1)
        plt.bar(range(1, len(accuracies) + 1), accuracies, color='blue', alpha=0.7)
        plt.xlabel('Fold Number')
        plt.ylabel('Accuracy')
        plt.title('Cross-Validation Accuracy for Each Fold')
        plt.ylim(0.8, 1.0)
        for i in range(len(accuracies)):
            plt.text(i + 1, accuracies[i], f'{accuracies[i]:.4f}', ha='center', va='bottom')

        # F1分数
        plt.subplot(1, 3, 2)
        plt.bar(range(1, len(f1_scores) + 1), f1_scores, color='green', alpha=0.7)
        plt.xlabel('Fold Number')
        plt.ylabel('F1 Score')
        plt.title('Cross-Validation F1 Score for Each Fold')
        plt.ylim(0.8, 1.0)
        for i in range(len(f1_scores)):
            plt.text(i + 1, f1_scores[i], f'{f1_scores[i]:.4f}', ha='center', va='bottom')

        # 自定义评分
        plt.subplot(1, 3, 3)
        plt.bar(range(1, len(custom_scores) + 1), custom_scores, color='purple', alpha=0.7)
        plt.xlabel('Fold Number')
        plt.ylabel('Custom Score')
        plt.title('Cross-Validation Custom Score for Each Fold')
        plt.ylim(0.8, 1.0)
        for i in range(len(custom_scores)):
            plt.text(i + 1, custom_scores[i], f'{custom_scores[i]:.4f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

# 使用方法
# model = RandomForestModel(data_path='训练集.xlsx')
# model.tune_hyperparameters()
# model.cross_validate()

# Best parameters found:  {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_split': 5, 'n_estimators': 100}
# Accuracy scores: [0.994535519125683, 1.0, 0.9890710382513661, 1.0, 0.9835164835164835, 1.0, 0.989010989010989, 1.0, 0.9835164835164835, 1.0]
# Mean accuracy: 0.9940
# Standard deviation: 0.0067
# F1 scores: [0.994510453805217, 1.0, 0.989181298972451, 1.0, 0.9835791175117018, 1.0, 0.9888865109453344, 1.0, 0.983826804424811, 1.0]
# Mean F1 score: 0.9940
# Standard deviation: 0.0066
# Custom scores (0.5*Acc + 0.5*F1): [0.99452298646545, 1.0, 0.9891261686119086, 1.0, 0.9835478005140926, 1.0, 0.9889487499781617, 1.0, 0.9836716439706472, 1.0]
# Mean custom score: 0.9940
# Standard deviation: 0.0067
