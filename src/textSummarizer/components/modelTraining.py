# modelTraining.py
import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer
)
from pathlib import Path
from textSummarizer.logger import logging
from textSummarizer.exception import CustomException

import sys

# ================================
# 1️⃣ Configuration Class
# ================================
class modelTrainingConfig:
    def __init__(self):
        # Paths
        self.output_dir = Path("artifacts/modelTraining")  # path to save trained model
        self.data_path = Path("artifacts/dataTransformation")  # path to tokenized dataset
        
        # Model
        self.model_ckpt = "facebook/bart-base"

        # Training hyperparameters
        self.num_train_epochs = 3
        self.learning_rate = 2e-5
        self.per_device_train_batch_size = 4
        self.per_device_eval_batch_size = 1
        self.gradient_accumulation_steps = 8
        self.eval_strategy = "steps"
        self.eval_steps = 1000
        self.logging_steps = 50
        self.save_strategy = "steps"
        self.save_steps = 1000
        self.save_total_limit = 2
        self.fp16 = True
        self.torch_compile = hasattr(torch, "compile")  # only enable if available


# ================================
# 2️⃣ Model Training Class
# ================================
class ModelTraining:
    def __init__(self):
        try:
            self.config = modelTrainingConfig()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logging.info(f"Using device: {self.device}")

            # Load tokenizer & model
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(self.device)

            # Load dataset
            self.dataset = load_from_disk(self.config.data_path)
            if "train" not in self.dataset or "validation" not in self.dataset:
                raise CustomException("Dataset must contain 'train' and 'validation' splits")

            # Data collator
            self.data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
                padding="longest"
            )

            os.makedirs(self.config.output_dir, exist_ok=True)
        except Exception as e:
            raise CustomException(f"Error in ModelTraining initialization: {e}", sys) from e

    def train(self):
        try:
            logging.info(f"Training on device: {self.device}")

            # Training arguments
            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_train_epochs,
                learning_rate=self.config.learning_rate,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                per_device_eval_batch_size=self.config.per_device_eval_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                eval_strategy=self.config.eval_strategy,
                eval_steps=self.config.eval_steps,
                logging_steps=self.config.logging_steps,
                save_strategy=self.config.save_strategy,
                save_steps=self.config.save_steps,
                save_total_limit=self.config.save_total_limit,
                fp16=self.config.fp16 and torch.cuda.is_available(),
                torch_compile=self.config.torch_compile,
                report_to="none"
            )

            # Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset['train'],
                eval_dataset=self.dataset['validation'],
                tokenizer=self.tokenizer,
                data_collator=self.data_collator
            )

            # Start training
            trainer.train()

            # Save model & tokenizer
            model_save_path = os.path.join(self.config.output_dir, "model")
            tokenizer_save_path = os.path.join(self.config.output_dir, "tokenizer")

            self.model.save_pretrained(model_save_path)
            self.tokenizer.save_pretrained(tokenizer_save_path)

            logging.info(f"Model saved at: {model_save_path}")
            logging.info(f"Tokenizer saved at: {tokenizer_save_path}")

            return model_save_path, tokenizer_save_path

        except Exception as e:
            raise CustomException(f"Error during training: {e}", sys) from e


# ================================
# 3️⃣ Main Execution
# ================================
if __name__ == "__main__":
    try:
        trainer_obj = ModelTraining()
        model_path, tokenizer_path = trainer_obj.train()
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise
