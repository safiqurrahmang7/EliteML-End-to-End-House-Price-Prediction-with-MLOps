import pandas as pd
from zenml import step
from src.handling_misisng_values import MissingValueHandler, dropped_missing_values, filling_missing_values



@step
def handling_missing_value_step(df:pd.DataFrame,strategy:str = 'mean'):

    if strategy == 'drop':
        handler = MissingValueHandler(dropped_missing_values(axis=1))
    elif strategy in ['mean','mode','median','constant']:
        handler = MissingValueHandler(filling_missing_values(method= strategy))
    else:
        raise ValueError(f"Unsupported Missing Value Strategy: {strategy} ")
    
    df_cleaned = handler.execute_handler(df)
    return df_cleaned




