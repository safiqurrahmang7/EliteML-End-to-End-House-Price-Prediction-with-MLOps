from src.feature_engineering import (
    log_transformation,
    MinMaxScaling,
    StandardScaling,
    OnhotEncoding,
    Featurengineer,
)
import pandas as pd
from zenml import step


@step
def feature_engineer_step(df: pd.DataFrame, method: str = 'standard_scaling', features: list = []) -> pd.DataFrame:
    if method == 'log':
        engineer = Featurengineer(log_transformation(features=features))
    elif method == 'standard_scaling':
        engineer = Featurengineer(StandardScaling(features=features))
    elif method == 'MinMax_Scaling':
        engineer = Featurengineer(MinMaxScaling(features=features))
    elif method == 'OneHot_Encoding':
        engineer = Featurengineer(OnhotEncoding(features=features))
    else:
        raise ValueError('Unsupported Feature Engineering')

    df_transformed = engineer.apply_feature_eng(df)
    
        

    
    return df_transformed

if __name__ == '__main__':

    path = 'D:\EliteML-End-to-End-House-Price-Prediction-with-MLOps\Extracted_Data\AmesHousing.csv'

    df = pd.read_csv(path)
    data = feature_engineer_step(df=df, method= 'log', features= ["Gr Liv Area", "SalePrice"])
    print(data)