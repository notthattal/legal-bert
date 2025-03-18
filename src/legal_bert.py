'''
This is the main script of the project. It calls the data processing pipeling, training pipeline, 
and evaluation pipeline. The main function in this script implements all of these steps sequentially. 
Furthermore, the main function is used to fine-tune the model using the script run_legal_bert.ipynb,
which calls this main function to start the training.
'''
import pandas as pd
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import wandb
from process_data import DataProcessor
from evaluate import BillClassifierEvaluator
from train import BillClassifierTrainer
import os


def main(run_evaluation=True):
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


    print('\nEvaluating the fine-tuned model...')
    
    if run_evaluation:
        # Initialize evaluator with default paths
        evaluator = BillClassifierEvaluator(
            base_model_name="distilbert-base-uncased",
            fine_tuned_model_path="../models/fine_tuned_legalbert",
            train_data_path="../data/masterCategorized.csv",
            test_data_path="../data/test.csv"
        )
        
        # Run evaluation on a small subset since it takes a couple of hours to evaluation on the entire set
        try:
            # Load data
            evaluator.load_data()
            
            # Load models
            evaluator.load_models()

            # Evaluate on a subset of training set and the full test set which is a small set 
            # created based on bills that are not included in the train set
            results = evaluator.compare_models(eval_on_train= True,
                                            eval_on_test=False,
                                            train_sample_size=2000,
                                            test_sample_size=25)
            
            evaluator.save_results(results,
                                output_dir='../evaluation_results')
            
            evaluator.generate_html_report(results,
                                        output_file='../evaluation_results/evaluation_report.html')
            
        except Exception as e:
            print(f"Error during evaluation test: {str(e)}")
            import traceback
            traceback.print_exc()

    

if __name__=='__main__':
    main()