from src.Outliter_Detection import IQROutlierDetection, ZscoreOutlierDetection, OulierDetector
from zenml import step
import pandas as pd



@step
def outlier_detection_step(df:pd.DataFrame,feature:str):


    df_numeric = df.select_dtypes(include = ['int','float'])

    outlier_detector = OulierDetector(ZscoreOutlierDetection(threshold=3))
    df_handled = outlier_detector.handle_outliers(df_numeric,method='remove')
    
    return df_handled
