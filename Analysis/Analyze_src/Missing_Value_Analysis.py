from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class MissingValuesAnalysis(ABC):

    def Analyze(self,df:pd.DataFrame):

        self.identifyMissingValues(df)
        self.VisualiseMissingValues(df)

    @abstractmethod
    def identifyMissingValues(self,df:pd.DataFrame):

        pass

    @abstractmethod
    def VisualiseMissingValues(self,df:pd.DataFrame):

        pass


class SimpleMissingValuesAnalysis(MissingValuesAnalysis):

    def identifyMissingValues(self, df: pd.DataFrame):
        
        Missing_values = df.isnull().sum()
        print(Missing_values[Missing_values>0])
    
    def VisualiseMissingValues(self, df: pd.DataFrame):

        plt.figure(figsize=(12,8))
        sns.heatmap(df.isnull(), cbar='False', cmap='viridis')
        plt.title("Missing Values Analysis")
        plt.show()

    