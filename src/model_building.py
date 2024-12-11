from abc import ABC, abstractmethod
import pandas as pd
import logging
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import RegressorMixin

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
class BuildModelStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self,xtrain:pd.DataFrame,ytrain:pd.Series) -> RegressorMixin:

        pass


class LinearModelStrategy(BuildModelStrategy):

    def build_and_train_model(self, xtrain, ytrain):
        
        if not isinstance(xtrain,pd.DataFrame):
            raise TypeError("xtrain must be a Pandas.DataFrame")
        if not isinstance(ytrain,pd.Series):
            raise TypeError("ytrain must be pandas.series")
        pipeline = Pipeline([
            ('Scaler',StandardScaler()),
            ('model',LinearRegression(fit_intercept=True))
        ])
        logging.info("Training The Linear Regression Model")
        pipeline.fit(xtrain,ytrain)
        logging.info("Model Trained")
        return pipeline
    

class ModelBuilder:

    def __init__(self,strategy:BuildModelStrategy):

        self.strategy = strategy

    def set_strategy(self,strategy:BuildModelStrategy):

        self.strategy = strategy
    
    def build_model(self,xtrain:pd.DataFrame,ytrain:pd.Series)-> RegressorMixin:

        logging.info("Building The Model")
        return self.strategy.build_and_train_model(xtrain,ytrain)
    
        


        