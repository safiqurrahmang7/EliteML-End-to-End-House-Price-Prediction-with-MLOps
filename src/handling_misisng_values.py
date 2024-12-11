from abc import ABC, abstractmethod
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format= "%(asctime)s - %(levelname)s - %(message)s")

class handling_missing_values(ABC):

    @abstractmethod
    def handle(self,df:pd.DataFrame)->pd.DataFrame:

        pass
        
class dropped_missing_values(handling_missing_values):

    def __init__(self,thresh:int = None,axis:int = 0):

        self._axis = axis
        self._thresh = thresh

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logging.info(f"Dropping the Missing values with axis: {self._axis} and thresh: {self._thresh}")
        datacleaned = df.dropna(axis=self._axis,thresh=self._thresh)
        logging.info(f"Missing Values Dropped")
        return datacleaned
    
class filling_missing_values(handling_missing_values):

    def __init__(self,method = 'mean', fill_value = None):

        self._method = method
        self._fill_value = fill_value

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logging.info(f"Filling The Missing Values With Method: {self._method}")
        df_cleaned = df.copy()

      
        
        if self._method == 'mean':
            numerical_features = df_cleaned.select_dtypes(include='number').columns
            df_cleaned[numerical_features] = df_cleaned[numerical_features].fillna(
                                df_cleaned[numerical_features].mean()   
                            )
        elif self._method == 'median':
                numerical_features = df_cleaned.select_dtypes(include='number').columns
                df_cleaned[numerical_features] = df_cleaned[numerical_features].fillna(
                                df_cleaned[numerical_features].median()   
                                )
        elif self._method == 'mode':
                numerical_features = df_cleaned.select_dtypes(include='number').columns
                df_cleaned[numerical_features] = df_cleaned[numerical_features].fillna(
                                df_cleaned[numerical_features].mode()   
                                )
        elif self._method == 'constant':
                df_cleaned = df_cleaned.fillna(self._fill_value)
        else:
             logging.warning(f"Unsknown Method {self._method}. No Missing Value Handled")

        logging.info("Missing values filled")

        return df_cleaned
    
class MissingValueHandler:
     
    def __init__(self,handler:handling_missing_values):
        
        self._handler = handler

    def set_handler(self,handler:handling_missing_values):
        
        self._handler = handler
        
    def execute_handler(self,df:pd.DataFrame):
         
        return self._handler.handle(df)

if __name__ == '__main__':
     
     path = 'D:/EliteML-End-to-End-House-Price-Prediction-with-MLOps/Extracted_Data/AmesHousing.csv'

     df = pd.read_csv(path)

     handler = MissingValueHandler(filling_missing_values())
     data_handler = handler.execute_handler(df)
     print(isinstance(data_handler,pd.DataFrame))

     

