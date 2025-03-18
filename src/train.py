import json
from sklearn.preprocessing import LabelEncoder
import wandb
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from process_data import DataProcessor
import pandas as pd
import os

class LegalDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    

class BillClassifierTrainer:
    '''
    A class to fine-tune distilbert on the different bills collected. 
    The goal is to allow for a more accurate prediction of a bill's subject given the bill's metadata.
    The code here was used to fine-tune the model on Colab (which took about ~3 hours on A100)
    '''

    def __init__(
        self, 
        base_model_name: str = "distilbert-base-uncased",
        train_data_path: str = "../data/masterCategorized.csv",
        device: str = None
    ):
        """
        Initialize the trainer with model paths and data path.
        
        Args:
            base_model_name: Name or path of the base model
            train_data_path: Path to the training data CSV
            device: Device to run the model on (None for auto-detection)
        """
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Store paths
        self.base_model_name = base_model_name
        self.train_data_path = train_data_path

        # Initialize tokenizers and models
        self.base_tokenizer = None
        self.base_model = None
        self.fine_tuned_tokenizer = None
        self.fine_tuned_model = None
        
        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        self.label_mapping = None
        
        # Initialize data
        self.train_data = None
        self.test_data = None
        self.unique_subjects = None

    def tokenize_texts(self, texts, tokenizer):
        return tokenizer(texts, padding=True, truncation=True, max_length=512)
    
    def encode_labels(self, df):
        encoded_labels = self.label_encoder.fit_transform(df)
        num_labels = len(self.label_encoder.classes_)

        label_mapping = dict(zip(range(len(self.label_encoder.classes_)), self.label_encoder.classes_))
        self.label_mapping = label_mapping
        wandb.log({"label_mapping": label_mapping})

        with open('../data/label_mapping.json', 'w') as f:
            json.dump(label_mapping, f)

        print("Label mapping saved to 'label_mapping.json'")

        return encoded_labels, num_labels
    
    def get_train_val_datasets(self, tokenizer, titles, encoded_labels):
        train_texts, val_texts, train_labels, val_labels = train_test_split(titles, encoded_labels, test_size=0.2, random_state=42)

        train_encodings = self.tokenize_texts(train_texts, tokenizer)
        val_encodings = self.tokenize_texts(val_texts, tokenizer)

        train_dataset = LegalDataset(train_encodings, train_labels)
        val_dataset = LegalDataset(val_encodings, val_labels)

        return train_dataset, val_dataset


    def train_model(self, model, tokenizer, train_dataset, val_dataset):
        training_args = TrainingArguments(
            output_dir='./results',
            eval_strategy='epoch',
            save_strategy='epoch',
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            load_best_model_at_end=True,
            learning_rate=5e-6,
            gradient_accumulation_steps=2,
            max_grad_norm=1.0,
            report_to="wandb",
            run_name="legal-bert-training"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        trainer.train()

        model.save_pretrained('./fine_tuned_legalbert')
        tokenizer.save_pretrained('./fine_tuned_legalbert')

        return trainer
        


def main():
    wandb.init(project="legal-bert-classification", name="training-run-1")

    # Initialization of data dir and csv file names
    DATA_DIR = '../data'
    UNCATEGORIZED_MASTER_CSV = 'masterUncategorized.csv'
    CATEGORIZED_MASTER_CSV = 'masterCategorized.csv'

    # Process data 
    data_processor = DataProcessor(DATA_DIR, UNCATEGORIZED_MASTER_CSV, CATEGORIZED_MASTER_CSV)
    # This line of code here have been commented since we have already processed the data which took 8+ hours
    # master_df = data_processor.process_data()

    master_df = pd.read_csv(os.path.join(DATA_DIR, CATEGORIZED_MASTER_CSV))
    categories = data_processor.get_categories(master_df)
    titles = data_processor.get_titles(master_df)


    bill_classifier_trainer = BillClassifierTrainer(base_model_name="distilbert-base-uncased",
                                                    train_data_path=os.path.join(DATA_DIR, CATEGORIZED_MASTER_CSV))
    
    model_name = "distilbert-base-uncased"
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(bill_classifier_trainer.device)
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)


    encoded_labels, num_labels = bill_classifier_trainer.encode_labels(categories)


    train_dataset, val_dataset = bill_classifier_trainer.get_train_val_datasets(tokenizer, titles, encoded_labels)

    trainer = bill_classifier_trainer.train_model(model, tokenizer, train_dataset, val_dataset)

    wandb.finish()


if __name__ == '__main__':
    main()
       


