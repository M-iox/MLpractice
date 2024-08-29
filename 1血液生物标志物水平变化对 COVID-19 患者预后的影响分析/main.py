from classification import RandomForestModel
from regression import RandomForestRegressionModel

ClassficationModel = RandomForestModel(data_path='训练集.xlsx')
ClassficationModel.tune_hyperparameters()
ClassficationModel.cross_validate()

RegressionModel = RandomForestRegressionModel(data_path='训练集.xlsx')
RegressionModel.tune_hyperparameters()
RegressionModel.cross_validate()