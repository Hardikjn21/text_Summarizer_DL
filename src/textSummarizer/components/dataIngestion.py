from dataclasses import dataclass
from pathlib import Path
from textSummarizer.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from textSummarizer.exception import CustomException
import sys

@dataclass
class DataIngestionConfig:
    raw_data_path: Path = Path('artifacts') / 'source_data' / 'data.csv'
    root_dir: Path = Path('artifacts') / 'dataIngestion'
    train_data_path: Path = root_dir / 'train.csv'
    val_data_path: Path = root_dir / 'val.csv'
    test_data_path: Path = root_dir / 'test.csv'
    

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()        

    def initiate_data_ingestion(self):
        #log we have entered
        logging.info("Entered the data ingestion method or component")
            
        try:
            # Create required directory
            self.ingestion_config.root_dir.mkdir(parents=True, exist_ok=True)
            logging.info("Created the data ingestion directory")

            # Read source data
            df = pd.read_csv(self.ingestion_config.raw_data_path)
            logging.info("Read the raw data as dataframe")

            # Split data: train + temp
            train_df, temp_df = train_test_split(
                df, test_size=0.3, random_state=42
            )

            # Split temp into val + test
            val_df, test_df = train_test_split(
                temp_df, test_size=0.5, random_state=42
            )
            logging.info("Split the data into train and test and val")  

            # Save splits
            train_df.to_csv(self.ingestion_config.train_data_path, index=False)
            val_df.to_csv(self.ingestion_config.val_data_path, index=False)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Saved the train, val and test data into data ingestion folder")

            return {
                "train_path": self.ingestion_config.train_data_path,
                "val_path": self.ingestion_config.val_data_path,
                "test_path": self.ingestion_config.test_data_path,
            }
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_path, val_path, test_path = obj.initiate_data_ingestion()
    logging.info(
        f"Data ingestion completed successfully and train, val, test files are created at {train_path}, {val_path}, {test_path}"
    )