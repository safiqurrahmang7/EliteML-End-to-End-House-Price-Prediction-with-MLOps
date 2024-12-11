from abc import ABC, abstractmethod
import logging
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.base import RegressorMixin

logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s")

class ModelEvaluatingStrategy(ABC):

    @abstractmethod
    def Model_evaluating(self,model:RegressorMixin,xtest:pd.DataFrame,ytest:pd.Series):
        
        pass

class LinearRegressionEvaluating(ModelEvaluatingStrategy):

    def Model_evaluating(self, model, xtest, ytest):
        

        
        ypred = model.predict(xtest)

        scores = {'mse':mean_squared_error(ytest,ypred),
                  'r2Score':r2_score(ytest,ypred)}
        logging.info(f'The metrics: {scores}')

        return scores
    

class ModelEvaluator:

    def __init__(self,strategy:ModelEvaluatingStrategy):

        self.strategy = strategy
    
    def set_strategy(self,strategy:ModelEvaluatingStrategy):

        self.strategy = strategy
    
    def evaluate(self,model,xtest,ytest):

        self.strategy.Model_evaluating(model,xtest,ytest)


        