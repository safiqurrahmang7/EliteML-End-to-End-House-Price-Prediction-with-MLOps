from src.model_evaluating import ModelEvaluator,LinearRegressionEvaluating
from sklearn.pipeline import Pipeline
import pandas as pd
from zenml import step

@step(enable_cache=False)
def Model_Evaluation_step(trained_model:Pipeline,xtest:pd.DataFrame,ytest:pd.Series):


    xtest = trained_model.named_steps['Preprocessor'].transform(xtest)

    evaluator = ModelEvaluator(LinearRegressionEvaluating())
    scores = evaluator.evaluate(trained_model.named_steps['model'],xtest,ytest)

    return scores