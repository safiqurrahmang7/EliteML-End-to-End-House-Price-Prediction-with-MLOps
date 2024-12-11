from src.data_splitter import splitter, TrainTestSplitter
import pandas as pd
from zenml import step
from typing import Tuple

@step
def data_splitter_step(df:pd.DataFrame,target:str)-> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    Data_splitter = splitter(TrainTestSplitter(test_size=0.2,random_state=42))

    xtrain,xtest,ytrain,ytest = Data_splitter.execute_splitter(df = df,target=target)

    return xtrain,xtest,ytrain,ytest