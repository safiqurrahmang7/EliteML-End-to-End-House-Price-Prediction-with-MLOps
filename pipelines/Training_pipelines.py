from zenml import pipeline, Model
from steps.ingest_data_step import ingest_data_step
from steps.handling_missing_values_step import handling_missing_value_step
from steps.feature_engineering_step import feature_engineer_step
from steps.outlier_detection_step import outlier_detection_step
from steps.Data_splitter_step import data_splitter_step
from steps.ModelBuildingStep import model_building_step
from steps.ModelEvaluationStep import Model_Evaluation_step

import pandas as pd


@pipeline(
         model=Model(
        # The name uniquely identifies this model
        name="prices_predictor"
    ),
)
def ml_pipepline():

    raw_data = ingest_data_step(file_path = 'D:/EliteML-End-to-End-House-Price-Prediction-with-MLOps/Data/archive.zip' )


    cleaned_data = handling_missing_value_step(df = raw_data,strategy = 'mean')
    

    transformed_data = feature_engineer_step(df = cleaned_data,method = 'standard_scaling', 
                                             features = ["Gr Liv Area", "SalePrice"])
    
    
    handled_data = outlier_detection_step(df = transformed_data, feature = "SalePrice" )
    

    xtrain,xtest,ytrain,ytest = data_splitter_step(df = handled_data,target = 'SalePrice')

    model = model_building_step(xtrain,ytrain)

    Evaluation_metrics = Model_Evaluation_step(trained_model = model,xtest = xtest, ytest = ytest)

    return model


if __name__ == '__main__':

    run = ml_pipepline()
    
