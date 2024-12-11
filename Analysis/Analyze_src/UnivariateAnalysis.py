import matplotlib.pyplot as plt
import pandas as pd
from abc import ABC, abstractmethod
import seaborn as sns

class UnivariateDataAnalysis(ABC):

    @abstractmethod
    def Analyze(self,df:pd.DataFrame,feature:str):

        pass


class NumericalDataAnalysis(UnivariateDataAnalysis):

    def Analyze(self, df: pd.DataFrame, feature: str):
       
       sns.histplot(df[feature], kde=True, bins = 30)
       plt.title(f"Distribution of {feature}")
       plt.show()

class CategoricalDataAnalysis(UnivariateDataAnalysis):
    
    def Analyze(self, df: pd.DataFrame, feature: str):
        
        plt.figure(figsize=(10,6))
        sns.countplot(x= feature, data=df, palette='muted',legend=False)
        plt.title(f'Distribution of {feature}')
        plt.xticks(rotation = 90)
        plt.show()


class DataAnalyzer:

    def __init__(self,analysis:UnivariateDataAnalysis):
        
        self._analysis = analysis

    def set_analysis(self,analysis:UnivariateDataAnalysis):

        self._analysis = analysis

    def execute_analysis(self,df:pd.DataFrame,feature:str):

        self._analysis.Analyze(df,feature)