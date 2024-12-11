from zenml import step
import pandas as pd
from src.ingest_data import ingest_hub


@step
def ingest_data_step(file_path:str):

    file_extenstion = '.zip'

    data_ingestor = ingest_hub().get_data_ingester(file_extenstion)

    df_ingested = data_ingestor.ingest(file_path)

    return df_ingested


    