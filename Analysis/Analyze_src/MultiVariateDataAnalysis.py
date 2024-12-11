from abc import ABC, abstractmethod
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class MultiVariateAnalyzer(ABC):

    def Analyze(self,df:pd.DataFrame):

        self.generate_correlation_heatmap(df)
        self.generate_paiplot(df)

    @abstractmethod
    def generate_correlation_heatmap(self,df:pd.DataFrame):

        pass

    @abstractmethod
    def generate_paiplot(self,df:pd.DataFrame):

        pass

class SimpleMultivariateAnalysis(MultiVariateAnalyzer):

    def generate_correlation_heatmap(self, df: pd.DataFrame):
        
        plt.figure(figsize=(10,8))
        sns.heatmap(df.corr(),annot=True, cmap='coolwarm', fmt='.2f',linewidths=0.5)
        plt.title('Correlation Heatmap')
        plt.show()

    def generate_paiplot(self, df: pd.DataFrame):
        
        plt.figure(figsize=(10,8))
        sns.pairplot(df)
        plt.title("Pairplot of Selecated Features")
        plt.show()

