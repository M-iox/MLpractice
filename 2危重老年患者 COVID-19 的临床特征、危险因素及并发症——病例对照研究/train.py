from regression import RandomForestRegressionModel

RegressionModel = RandomForestRegressionModel(data_path='训练集.xlsx')
RegressionModel.tune_hyperparameters()
RegressionModel.cross_validate()