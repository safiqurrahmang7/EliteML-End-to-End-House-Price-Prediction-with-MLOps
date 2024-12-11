import numpy as np
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
import pandas as pd
import logging


class FeatureEngineeringBase(ABC):

    @abstractmethod
    def apply_tranformation(self,df:pd.DataFrame)->pd.DataFrame:

        pass

class log_transformation(FeatureEngineeringBase):

    def __init__(self,features):
        
        self._features = features
        
    def apply_tranformation(self, df: pd.DataFrame):
        if not isinstance(df,pd.DataFrame):
            raise ValueError("The df must be a pandas dataframe")

        logging.info(f"Applying Transformation to features: {self._features}")
        df_transformed = df

        for feature in self._features:

            df_transformed[feature] = np.log1p(
                df[feature]
                )
        logging.info("Log Transformation applied")
        return df_transformed

class MinMaxScaling(FeatureEngineeringBase):

    def __init__(self,features, feature_range= (0,1)):

        self._features = features
        self._scaler = MinMaxScaler(feature_range=feature_range)

    def apply_tranformation(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logging.info(f"Applying MinMaxScaling To The Features: {self._features}")
        df_transformed = df.copy()
        df_transformed[self._features] = self._scaler.fit_transform(df_transformed[self._features])
        logging.info("MinMax Scaling Applied")
        return df_transformed
    
class StandardScaling(FeatureEngineeringBase):

    def __init__(self,features):
        
        self._features = features
        self._scaler = StandardScaler()
    
    def apply_tranformation(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logging.info(f"Applying Standard Scaler To The Features: {self._features}")
        df_transformed = df.copy()
        df_transformed[self._features] = self._scaler.fit_transform(df_transformed[self._features])
        logging.info("Standard Scaler Applied")
        return df_transformed

class OnhotEncoding(FeatureEngineeringBase):

    def __init__(self,features):
        
        self._features = features
        self._encoder = OneHotEncoder(drop='first',sparse=False)

    def apply_tranformation(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logging.info(f"Applyinh OneHot Encoding To The Features: {self._features}")
        df_transformed = df.copy()

        df_encoded = pd.DataFrame(self._encoder.fit_transform(df_transformed[self._features]))
        df_transformed = df_transformed.drop(columns=self._features).reset_index(drop=True)
        df_transformed = pd.concat([df_transformed,df_encoded],axis=1)
        logging.info("OneHot Encoding Applied")

        return df_transformed
    
class Featurengineer:

    def __init__(self, strategy = FeatureEngineeringBase):
        
        self.strategy = strategy

    def set_feature_eng(self, strategy = FeatureEngineeringBase):
        
        self.strategy = strategy
    
    def apply_feature_eng(self,df:pd.DataFrame):

        return self.strategy.apply_tranformation(df)
    

if __name__ == '__main__':

    path = 'D:\EliteML-End-to-End-House-Price-Prediction-with-MLOps\Extracted_Data\AmesHousing.csv'

    df = pd.read_csv(path)
    engineer = Featurengineer(StandardScaling(["Gr Liv Area", "SalePrice"]))
    df_cleaned = engineer.apply_feature_eng(df)
    print(df_cleaned)
    