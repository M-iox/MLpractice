from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class RandomForestRegressionModel:
    def __init__(self, data_path, sheet_name=1):
        self.data_path = data_path
        self.sheet_name = sheet_name
        self.train_data = pd.read_excel(self.data_path, sheet_name=self.sheet_name)
        self.train_data.columns = self.train_data.columns.str.strip()
        self.features = self.train_data.iloc[:, 2:-1]
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.array(['No', 'Yes'])
        self.target = self.label_encoder.transform(self.train_data['Deceased'].values.ravel())
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
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return 0.5 * mae + 0.5 * rmse

    def tune_hyperparameters(self):
        # 超参数调优
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'max_features': ['auto', 'sqrt', 'log2']
        }

        grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error')
        grid_search.fit(self.features, self.target)
        print("Best parameters found: ", grid_search.best_params_)
        self.rf_model = RandomForestRegressor(**grid_search.best_params_, random_state=42)
        # self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    def cross_validate(self):
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        maes, rmses, custom_scores = [], [], []

        for train_index, test_index in kf.split(self.features):
            X_train_fold, X_test_fold = self.features.iloc[train_index], self.features.iloc[test_index]
            y_train_fold, y_test_fold = self.target[train_index], self.target[test_index]

            self.rf_model.fit(X_train_fold, y_train_fold)
            y_pred_fold = self.rf_model.predict(X_test_fold)

            mae = mean_absolute_error(y_test_fold, y_pred_fold)
            rmse = np.sqrt(mean_squared_error(y_test_fold, y_pred_fold))
            custom = self.custom_scorer(y_test_fold, y_pred_fold)

            maes.append(mae)
            rmses.append(rmse)
            custom_scores.append(custom)

        self._print_results(maes, rmses, custom_scores)
        self._plot_results(maes, rmses, custom_scores)

    def _print_results(self, maes, rmses, custom_scores):
        print(f'MAE scores: {maes}')
        print(f'Mean MAE: {np.mean(maes):.4f}')
        print(f'Standard deviation: {np.std(maes):.4f}')

        print(f'RMSE scores: {rmses}')
        print(f'Mean RMSE: {np.mean(rmses):.4f}')
        print(f'Standard deviation: {np.std(rmses):.4f}')

        print(f'Custom scores (0.5*MAE + 0.5*RMSE): {custom_scores}')
        print(f'Mean custom score: {np.mean(custom_scores):.4f}')
        print(f'Standard deviation: {np.std(custom_scores):.4f}')

    def _plot_results(self, maes, rmses, custom_scores):
        plt.figure(figsize=(15, 6))

        # MAE
        plt.subplot(1, 3, 1)
        plt.bar(range(1, len(maes) + 1), maes, color='blue', alpha=0.7)
        plt.xlabel('Fold Number')
        plt.ylabel('MAE')
        plt.title('Cross-Validation MAE for Each Fold')
        for i in range(len(maes)):
            plt.text(i + 1, maes[i], f'{maes[i]:.4f}', ha='center', va='bottom')

        # RMSE
        plt.subplot(1, 3, 2)
        plt.bar(range(1, len(rmses) + 1), rmses, color='green', alpha=0.7)
        plt.xlabel('Fold Number')
        plt.ylabel('RMSE')
        plt.title('Cross-Validation RMSE for Each Fold')
        for i in range(len(rmses)):
            plt.text(i + 1, rmses[i], f'{rmses[i]:.4f}', ha='center', va='bottom')

        # 自定义评分
        plt.subplot(1, 3, 3)
        plt.bar(range(1, len(custom_scores) + 1), custom_scores, color='purple', alpha=0.7)
        plt.xlabel('Fold Number')
        plt.ylabel('Custom Score')
        plt.title('Cross-Validation Custom Score for Each Fold')
        for i in range(len(custom_scores)):
            plt.text(i + 1, custom_scores[i], f'{custom_scores[i]:.4f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

# 使用方法
model = RandomForestRegressionModel(data_path='训练集.xlsx')
model.tune_hyperparameters()
model.cross_validate()
