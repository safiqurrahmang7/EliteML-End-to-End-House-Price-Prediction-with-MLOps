from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
import pandas as pd

import logging



class DataSplitterBase(ABC):

    @abstractmethod
    def split(self,df:pd.DataFrame,target:str):

        pass


class TrainTestSplitter(DataSplitterBase):

    def  __init__(self,test_size = 0.2,random_state = 42):

        self.test_size = test_size
        self.random_state = random_state
    
    def split(self, df: pd.DataFrame, target: str):
        
        x = df.drop(columns=target)
        y = df[target]

        xtrain,xtest,ytrain,ytest = train_test_split(x,y,
                                test_size=self.test_size,
                                random_state=self.random_state)
        categorical_col = xtrain.select_dtypes(include=['object','category']).columns
        logging.info(f"Xtrain columsn are: {categorical_col}")
        
        return xtrain,xtest,ytrain,ytest
    
class splitter:

    def __init__(self,strategy:DataSplitterBase) -> None:
        
        self.strategy = strategy
    
    def set_strategy(self,strategy:DataSplitterBase):

        self.strategy = strategy
    
    def execute_splitter(self,df:pd.DataFrame,target:str):
        
        return self.strategy.split(df,target)
        


