import pandas as pd
from abc import ABC, abstractmethod


class DataInspectionStrategy(ABC):

    @abstractmethod
    def inspect(self,df:pd.DataFrame):
        pass

class DataTypeInspectionStrategy(DataInspectionStrategy):

    def inspect(self, df: pd.DataFrame):
        
        print(df.info())

class SummaryInspectionStrategy(DataInspectionStrategy):

    def inspect(self, df: pd.DataFrame):
        
        print("\n Summary Statistics of Numerical Features")
        print(df.describe())

        print("\n Summary Statistics of Categorical Features")
        print(df.describe(include=['object']))

class DataInspector:

    def __init__(self,strategy:DataInspectionStrategy):
        
        self._strategy = strategy

    def set_strategy(self,strategy:DataInspectionStrategy):

        self._strategy = strategy
    
    def execute_strategy(self,df:pd.DataFrame):

        self._strategy.inspect(df)