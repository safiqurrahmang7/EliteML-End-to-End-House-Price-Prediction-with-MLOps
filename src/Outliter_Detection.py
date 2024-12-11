from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from steps.feature_engineering_step import feature_engineer_step
from steps.handling_missing_values_step import handling_missing_value_step

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class OutlierDetectionBase(ABC):

    @abstractmethod
    def DetectOutlier(self,df:pd.DataFrame)->pd.DataFrame:

        pass

class ZscoreOutlierDetection(OutlierDetectionBase):

    def __init__(self, threshold:int = 3):

        self._threshold = threshold
        

    def DetectOutlier(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logging.info(f"Detecting zscore Outliers with the threshold: {self._threshold}")
        df_numeric = df.select_dtypes(include='number')
        zscore = np.abs((df_numeric-df_numeric.mean())/df_numeric.std())
        outliers = zscore > self._threshold
        logging.info("Outlier Detected")
        return outliers

class IQROutlierDetection(OutlierDetectionBase):

    def DetectOutlier(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logging.info("Detecting The IQR Based Outlier")
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3-Q1
        outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
        logging.info("Outlier Detected")
        return outliers
    
class OulierDetector:

    def __init__(self, strategy = OutlierDetectionBase):

        self.strategy = strategy

    def set_strategy(self, strategy = OutlierDetectionBase):

        self.strategy = strategy
    
    def detect_outliers(self,df:pd.DataFrame):

        
        return self.strategy.DetectOutlier(df)
    
    def handle_outliers(self,df:pd.DataFrame, method:str = 'remove'):

        outliers = self.detect_outliers(df)

        if method == 'remove':
            df_cleaned = df[(~outliers).all(axis=1)]
            return df_cleaned
        elif method == 'cap':
            df_cleaned = df.clip(lower=df.quantile(0.01),upper=df.quantile(0.99),axis = 1)
            return df_cleaned
        else:
            logging.warning(f"Unknown Mehtod: {method}. No Outlier Handled")
            return df
        
    def visulizeOutlier(self,df:pd.DataFrame,feature:str):
        
        plt.figure(figsize=(8,6))
        sns.boxplot(x = df[feature])
        plt.show()
        
