import pandas as pd
import gc
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from dataclasses import dataclass
from pathlib import Path
import os
import sys

from textSummarizer.logger import logging
from textSummarizer.exception import CustomException

# ======================================================
# CONFIG
# ======================================================
@dataclass
class DataTransformationConfig:
    train_csv_path: Path = Path("artifacts/dataIngestion/train.csv")
    val_csv_path: Path = Path("artifacts/dataIngestion/val.csv")
    test_csv_path: Path = Path("artifacts/dataIngestion/test.csv")

    transformed_data_dir: Path = Path("artifacts/dataTransformation")

    model_name: str = "facebook/bart-base"
    max_input_length: int = 1024
    max_target_length: int = 128


# ======================================================
# DATA TRANSFORMATION
# ======================================================
class DataTransformation:
    def __init__(self):
        try:
            logging.info("Initializing DataTransformation")

            self.config = DataTransformationConfig()
            os.makedirs(self.config.transformed_data_dir, exist_ok=True)

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name
            )

            logging.info("Tokenizer loaded successfully")

        except Exception as e:
            logging.error("Error during DataTransformation initialization")
            raise CustomException(e, sys)

    # --------------------------------------------------
    def _load_csv(self, path: Path) -> pd.DataFrame:
        try:
            logging.info(f"Loading CSV file from {path}")
            return pd.read_csv(
                path,
                usecols=["article", "highlights"],
            )
        except Exception as e:
            raise CustomException(e, sys)

    # --------------------------------------------------
    def _clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Cleaning dataframe")

            df = df.rename(
                columns={
                    "article": "input_text",
                    "highlights": "target_text"
                }
            )

            df["input_text"] = (
                df["input_text"]
                .fillna("")
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .str.replace("\u200b", "", regex=False)
            )

            df["target_text"] = (
                df["target_text"]
                .fillna("")
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .str.replace("\u200b", "", regex=False)
            )

            return df

        except Exception as e:
            raise CustomException(e, sys)

    # --------------------------------------------------
    def _create_hf_dataset(self) -> DatasetDict:
        try:
            logging.info("Creating HuggingFace Dataset")

            dataset = DatasetDict()

            train_df = self._clean_df(self._load_csv(self.config.train_csv_path))
            dataset["train"] = Dataset.from_pandas(train_df, preserve_index=False)
            del train_df
            gc.collect()

            val_df = self._clean_df(self._load_csv(self.config.val_csv_path))
            dataset["validation"] = Dataset.from_pandas(val_df, preserve_index=False)
            del val_df
            gc.collect()

            test_df = self._clean_df(self._load_csv(self.config.test_csv_path))
            dataset["test"] = Dataset.from_pandas(test_df, preserve_index=False)
            del test_df
            gc.collect()

            logging.info("HF Dataset creation completed")
            return dataset

        except Exception as e:
            raise CustomException(e, sys)

    # --------------------------------------------------
    def _tokenize_function(self, batch):
        try:
            model_inputs = self.tokenizer(
                batch["input_text"],
                truncation=True,
                max_length=self.config.max_input_length,
                padding=False,
            )

            labels = self.tokenizer(
                text_target=batch["target_text"],
                truncation=True,
                max_length=self.config.max_target_length,
                padding=False,
            )

            labels["input_ids"] = [
                [(t if t != self.tokenizer.pad_token_id else -100) for t in seq]
                for seq in labels["input_ids"]
            ]

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        except Exception as e:
            raise CustomException(e, sys)

    # --------------------------------------------------
    def initiate_data_transformation(self):
        try:
            logging.info("Starting data transformation pipeline")

            dataset = self._create_hf_dataset()

            tokenized_dataset = dataset.map(
                self._tokenize_function,
                batched=True,
                batch_size=128,
                remove_columns=["input_text", "target_text"],
            )

            tokenized_dataset.save_to_disk(
                self.config.transformed_data_dir
            )

            logging.info("Data transformation completed successfully")

            return self.config.transformed_data_dir

        except Exception as e:
            logging.error("Data transformation failed")
            raise CustomException(e, sys)


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    try:
        data_transformation = DataTransformation()
        data_transformation.initiate_data_transformation()
        print("Data Transformation Completed!")
    except Exception as e:
        logging.exception("Fatal error in data transformation")
        raise
