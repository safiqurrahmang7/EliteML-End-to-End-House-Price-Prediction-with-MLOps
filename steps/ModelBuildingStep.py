from zenml import Model,step,ArtifactConfig
from zenml.client import Client
import pandas as pd
from typing import Annotated
from sklearn.pipeline import Pipeline
import logging
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import mlflow
from xgboost import XGBRegressor

model = Model(
    name = 'Price_predictor',
    version = None,
    license = 'Apache 2.0',
    description = 'Price_predictor for House'
)

experiment_tracker = Client().active_stack.experiment_tracker

@step(enable_cache=False, experiment_tracker = experiment_tracker.name, model=model)
def model_building_step(xtrain:pd.DataFrame,ytrain:pd.Series)->Annotated[Pipeline, ArtifactConfig(name='skleran_pipeline',is_model_artifact=True)]:


    if not isinstance(xtrain,pd.DataFrame):
        raise TypeError("Xtrain must be a pandas.DataFrame")
    elif not isinstance(ytrain,pd.Series):
        raise TypeError("ytrain must be a pandas.Series")
    
    categorical_col = xtrain.select_dtypes(include=['object','category']).columns
    numerical_col = xtrain.select_dtypes(exclude=['object','category']).columns
    xtrain[numerical_col] = xtrain[numerical_col].astype('float64')

    logging.info(f"Categorical_Column: {categorical_col}")
    logging.info(f"Numerical_Column: {numerical_col}")

    numerical_transformer = SimpleImputer(strategy='mean')
    categorical_transformer = Pipeline(
        steps=[
            ('impute',SimpleImputer(strategy='most_frequent')),
            ('One-Hot',OneHotEncoder(handle_unknown='ignore'))
        ]
    )

    Preprocessor = ColumnTransformer(
            transformers=[
                ('num',numerical_transformer,numerical_col),
                ('cat',categorical_transformer,categorical_col)
            ]
    )

    
    
    pipeline = Pipeline(
        steps=[
            ('Preprocessor',Preprocessor),
            ('model',XGBRegressor())
        ]
    )

    if not mlflow.active_run():
        mlflow.start_run()

    try:
        mlflow.autolog()
        pipeline.fit(xtrain,ytrain)

        OneHot_Encoder =( pipeline.named_steps['Preprocessor'].named_transformers_['cat'].named_steps['One-Hot'])

        OneHot_Encoder.fit(xtrain[categorical_col])
        Expected_columns = numerical_col.tolist()+list(OneHot_Encoder.get_feature_names_out(categorical_col))
        logging.info(f"These are the Expected columns: {Expected_columns}")

    except Exception as e:
        logging.info('Error during trainng the model')
        raise e
    
    finally:
        mlflow.end_run()

    return pipeline


