from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class BivariateDataAnalysis(ABC):

    @abstractmethod
    def analyze(self,df:pd.DataFrame,feature1:str,feature2:str):

        pass

class NumericalvsNumericalAnalysis(BivariateDataAnalysis):

    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        
        
        plt.figure(figsize=(8,6))
        sns.scatterplot(x = feature1, y= feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()

class CategoricalvsNumerical(BivariateDataAnalysis):

    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        
        
        plt.figure(figsize=(10,6))
        sns.boxplot(x = feature1, y= feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.xticks(rotation = 45)
        plt.show()

class BivariateAnalyzer:

    def __init__(self,analyzer:BivariateDataAnalysis):

        self._analyzer = analyzer

    def set_analyzer(self,analyzer:BivariateDataAnalysis):

        self._analyzer = analyzer

    def ExecuteAnalyzer(self,df:pd.DataFrame,feature1:str,feature2:str):

        self._analyzer.analyze(df,feature1,feature2)