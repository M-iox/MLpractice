from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN  # 替换SMOTE为ADASYN
from collections import Counter
from sklearn.preprocessing import MinMaxScaler

class RandomForestModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.train_data = pd.read_csv(self.data_path)
        self.train_data.columns = self.train_data.columns.str.strip()
        self.features = self.train_data.iloc[:, 2:-1]
        self.features.drop(columns=['IL-6'], inplace=True)
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.array(['Control','COVID-19'])
        self.labels = self.label_encoder.transform(self.train_data['Group'].values.ravel())

    def process_data(self, X):
        numeric_features = X.dtypes[X.dtypes != 'object'].index
        categorical_features = X.columns.difference(numeric_features)

        numeric_imputer = SimpleImputer(strategy='median')
        categorical_imputer = SimpleImputer(strategy='most_frequent')

        X.loc[:, numeric_features] = numeric_imputer.fit_transform(X.loc[:, numeric_features])
        X.loc[:, categorical_features] = categorical_imputer.fit_transform(X.loc[:, categorical_features])

        # One-Hot编码处理离散值
        X = pd.get_dummies(X, dummy_na=True, dtype=int)

        # 对数值型特征进行标准化
        numeric_features = X.dtypes[X.dtypes != 'object'].index  # 重新获取标准化后的数值特征
        scaler = MinMaxScaler()
        X.loc[:, numeric_features] = scaler.fit_transform(X.loc[:, numeric_features])


        return X

    def custom_scorer(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        return 0.5 * acc + 0.5 * f1

    def tune_hyperparameters(self):
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
        self.rf_model = RandomForestClassifier(
            max_depth=10,
            max_features='sqrt',
            min_samples_split=5,
            n_estimators=100,
            random_state=42
        )

    def cross_validate(self):
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        accuracies, f1_scores, custom_scores = [], [], []

        smote = ADASYN(random_state=42)

        for train_index, test_index in kf.split(self.features):
            X_train_fold, X_test_fold = self.features.iloc[train_index], self.features.iloc[test_index]
            y_train_fold, y_test_fold = self.labels[train_index], self.labels[test_index]

            # 在训练集上进行特征处理
            X_train_fold = self.process_data(X_train_fold)
            X_test_fold = self.process_data(X_test_fold)
            X_test_fold = X_test_fold.reindex(columns=X_train_fold.columns, fill_value=0)

            # 应用SMOTE来平衡训练数据
            X_train_fold, y_train_fold = smote.fit_resample(X_train_fold, y_train_fold)

            # 输出SMOTE后的样本分布
            print(f"Fold {kf.get_n_splits()}: Resampled class distribution: {Counter(y_train_fold)}")

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

    def predict_test_set(self, test_data_path, output_csv):
        test_data = pd.read_csv(test_data_path)
        test_data.columns = test_data.columns.str.strip()
        self.test_features = test_data.iloc[:, 1:-1]
        self.test_features.drop(columns=['IL-6'], inplace=True)

        self.test_features = self.process_data(self.test_features)
        predictions = self.rf_model.predict(self.test_features)

        probabilities = self.rf_model.predict_proba(self.test_features)
        print(probabilities)

        predictions = (probabilities[:, 1] > 0.5).astype(int)

        results = pd.DataFrame({'id': test_data.iloc[:, 0], 'Group': predictions})
        results.to_csv(output_csv, index=False)

# 使用方法
model = RandomForestModel(data_path='train.csv')
model.default_parameters()
model.cross_validate()
model.predict_test_set(test_data_path='test.csv', output_csv='预测结果.csv')
print(model.features)
print(model.test_features)
print(model.labels)