from abc import ABC, abstractmethod
import pandas as pd
import zipfile
import os
class DataIngestor(ABC):

    @abstractmethod
    def ingest(self,file_path:str)-> pd.DataFrame:
        pass

class Zip_DataIngestor(DataIngestor):

    def ingest(self, file_path: str) -> pd.DataFrame:

        if not file_path.endswith('.zip'):
            raise ValueError("The Provided file is not .zip file")
        
        with zipfile.ZipFile(file_path,'r')as zip_ref:
            zip_ref.extractall('Extracted_Data')

        files = os.listdir('Extracted_Data')
        csv_files = [f for f in files if f.endswith('csv')]

        if len(csv_files) == 0:
            raise FileNotFoundError("No CSV file found in the extracted data folder")
        elif len(csv_files)>1:
            raise ValueError("Multiplt CSV Files Found. Please Choose the Right One")

        csv_file_path = os.path.join('Extracted_data',csv_files[0])
        df = pd.read_csv(csv_file_path)
        return df

class ingest_hub:

        @staticmethod
        def get_data_ingester(file_extension:str)->DataIngestor:
            if file_extension == '.zip':
                return Zip_DataIngestor()
            else:
                raise ValueError(f"No DataIngester is not found for the given extension: {file_extension}")


