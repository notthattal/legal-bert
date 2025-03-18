import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.preprocessing import LabelEncoder
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from bert_score import BERTScorer
import nltk
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Union

# Download required NLTK packages if not already downloaded
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')


class BillClassifierEvaluator:
    """
    A class to evaluate the performance of bill classification models.
    Supports both base and fine-tuned models, with comprehensive metrics.
    """
    
    def __init__(
        self, 
        base_model_name: str = "distilbert-base-uncased",
        fine_tuned_model_path: str = "../models/fine_tuned_legalbert",
        train_data_path: str = "../data/masterCategorized.csv",
        test_data_path: str = "../data/test.csv",
        device: str = None
    ):
        """
        Initialize the evaluator with model paths and data paths.
        
        Args:
            base_model_name: Name or path of the base model
            fine_tuned_model_path: Path to the fine-tuned model
            train_data_path: Path to the training data CSV
            test_data_path: Path to the test data CSV
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
        self.fine_tuned_model_path = fine_tuned_model_path
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        
        # Initialize tokenizers and models
        self.base_tokenizer = None
        self.base_model = None
        self.fine_tuned_tokenizer = None
        self.fine_tuned_model = None
        
        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        with open("../data/label_mapping.json") as f:
            self.label_mapping = json.load(f)
        
        # Initialize data
        self.train_data = None
        self.test_data = None
        self.unique_subjects = None
        
        # Evaluation metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
        self.smooth_fn = SmoothingFunction().method1
        
    def load_data(self) -> None:
        """
        Load and preprocess the training and test data.
        """
        print("Loading and preprocessing data...")
        
        # Load train data
        self.train_data = pd.read_csv(self.train_data_path)
        
        # Load test data
        self.test_data = pd.read_csv(self.test_data_path)
        
        # Get unique subjects from both datasets
        all_subjects = pd.concat([self.train_data['Subject'], self.test_data['Subject']]).unique()
        self.unique_subjects = sorted(all_subjects)
        
        # Fit label encoder on all unique subjects
        self.label_encoder.fit(self.unique_subjects)
        
        print(f"Loaded {len(self.train_data)} training examples and {len(self.test_data)} test examples")
        print(f"Found {len(self.unique_subjects)} unique subjects")
        
    def load_models(self) -> None:
        """
        Load the base and fine-tuned models.
        """
        print("Loading models...")
        
        # Load base model and tokenizer
        print(f"Loading base model: {self.base_model_name}")
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=len(self.unique_subjects)
        ).to(self.device)
        
        # Load fine-tuned model and tokenizer
        print(f"Loading fine-tuned model from: {self.fine_tuned_model_path}")
        self.fine_tuned_tokenizer = AutoTokenizer.from_pretrained(self.fine_tuned_model_path)
        self.fine_tuned_model = AutoModelForSequenceClassification.from_pretrained(
            self.fine_tuned_model_path
        ).to(self.device)
        
        print("Models loaded successfully")
        
    def prepare_input_features(self, row: pd.Series) -> str:
        """
        Prepare input features from a data row.
        
        Args:
            row: A pandas Series containing bill information
            
        Returns:
            A string containing the formatted input features
        """
        # Combine relevant features into a single text
        return row['Title']
    
    def predict_subject(
        self, 
        model: Any, 
        tokenizer: Any, 
        input_text: str
    ) -> str:
        """
        Predict the subject of a bill using the given model.
        
        Args:
            model: The model to use for prediction
            tokenizer: The tokenizer for the model
            input_text: The input text containing bill features
            
        Returns:
            The predicted subject
        """
        # Tokenize input
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        ).to(self.device)
        
        # Get prediction
        if model == self.fine_tuned_model:
            with torch.no_grad():
                outputs = model(**inputs)
                pred = str(torch.argmax(outputs.logits, dim=1).item())
                predicted_subject = self.label_mapping[pred]
            
        else:
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class_idx = torch.argmax(logits, dim=1).item()
            
            # Decode class index to subject
            predicted_subject = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        
        return predicted_subject
    
    def evaluate_model(
        self, 
        model: Any, 
        tokenizer: Any, 
        data: pd.DataFrame, 
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Evaluate a model on the given data.
        
        Args:
            model: The model to evaluate
            tokenizer: The tokenizer for the model
            data: The data to evaluate on
            model_name: A name for the model (for reporting)
            
        Returns:
            A dictionary containing evaluation metrics
        """
        print(f"Evaluating {model_name}...")
        
        # Put model in evaluation mode
        model.eval()
        
        # Initialize result storage
        results = {
            'predictions': [],
            'ground_truth': [],
            'input_texts': [],
            'rouge1_f': [],
            'rouge2_f': [],
            'rougeL_f': [],
            'bleu': [],
            'bertscore_f1': []
        }
        
        # Process each row
        for idx, row in tqdm(data.iterrows(), total=len(data), desc=f"Evaluating {model_name}"):
            # Prepare input
            input_text = self.prepare_input_features(row)
            
            # Get ground truth
            ground_truth = row['Subject']
            
            # Get prediction
            prediction = self.predict_subject(model, tokenizer, input_text)
            
            # Store results
            results['predictions'].append(prediction)
            results['ground_truth'].append(ground_truth)
            results['input_texts'].append(input_text)
            
            # Calculate ROUGE scores
            rouge_scores = self.rouge_scorer.score(ground_truth, prediction)
            results['rouge1_f'].append(rouge_scores['rouge1'].fmeasure)
            results['rouge2_f'].append(rouge_scores['rouge2'].fmeasure)
            results['rougeL_f'].append(rouge_scores['rougeL'].fmeasure)
            
            # Calculate BLEU score
            reference_tokens = [nltk.word_tokenize(ground_truth.lower())]
            prediction_tokens = nltk.word_tokenize(prediction.lower())
            bleu = corpus_bleu([reference_tokens], [prediction_tokens], smoothing_function=self.smooth_fn)
            results['bleu'].append(bleu)
        
        # Calculate BERTScore (batch processing for efficiency)
        P, R, F1 = self.bert_scorer.score(results['predictions'], results['ground_truth'])
        results['bertscore_f1'] = F1.tolist()
        
        # Calculate classification metrics
        accuracy = accuracy_score(results['ground_truth'], results['predictions'])
        precision, recall, f1, _ = precision_recall_fscore_support(
            results['ground_truth'], 
            results['predictions'], 
            average='weighted',
            zero_division=0 
        )
        
        # Calculate average metrics
        avg_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_rouge1_f': np.mean(results['rouge1_f']),
            'avg_rouge2_f': np.mean(results['rouge2_f']),
            'avg_rougeL_f': np.mean(results['rougeL_f']),
            'avg_bleu': np.mean(results['bleu']),
            'avg_bertscore_f1': np.mean(results['bertscore_f1'])
        }
        
        # Create detailed classification report
        class_report = classification_report(
            results['ground_truth'], 
            results['predictions'],
            output_dict=True,
            zero_division=0
        )
        
        return {
            'model_name': model_name,
            'detailed_results': results,
            'avg_metrics': avg_metrics,
            'classification_report': class_report
        }
    
    def compare_models(
        self, 
        eval_on_train: bool = True, 
        eval_on_test: bool = True,
        train_sample_size: int = 25, 
        test_sample_size: int = 25,
    ) -> Dict[str, Any]:
        """
        Compare the base and fine-tuned models on a sample of the training and/or test data.
        
        Args:
            eval_on_train: Whether to evaluate on training data
            eval_on_test: Whether to evaluate on test data
            
        Returns:
            A dictionary containing evaluation results
        """
        results = {}
        
        sample_train = self.train_data.sample(train_sample_size)
        sample_test = self.test_data.sample(test_sample_size)

        # Evaluate on training data
        if eval_on_train:
            print("\nEvaluating models on training data...")
            base_train_results = self.evaluate_model(
                self.base_model, 
                self.base_tokenizer, 
                sample_train, 
                "Base Model (Train)"
            )
            
            fine_tuned_train_results = self.evaluate_model(
                self.fine_tuned_model, 
                self.fine_tuned_tokenizer, 
                sample_train, 
                "Fine-tuned Model (Train)"
            )
            
            results['train'] = {
                'base_model': base_train_results,
                'fine_tuned_model': fine_tuned_train_results
            }
        
        # Evaluate on test data
        if eval_on_test:
            print("\nEvaluating models on test data...")
            base_test_results = self.evaluate_model(
                self.base_model, 
                self.base_tokenizer, 
                sample_test, 
                "Base Model (Test)"
            )
            
            fine_tuned_test_results = self.evaluate_model(
                self.fine_tuned_model, 
                self.fine_tuned_tokenizer, 
                sample_test, 
                "Fine-tuned Model (Test)"
            )
            
            results['test'] = {
                'base_model': base_test_results,
                'fine_tuned_model': fine_tuned_test_results
            }
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "../evaluation_results") -> None:
        """
        Save evaluation results to files.
        
        Args:
            results: The evaluation results
            output_dir: Directory to save results
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON results
        json_path = os.path.join(output_dir, "evaluation_results.json")
        
        # Prepare a simplified version of results for JSON
        json_results = {}
        
        for data_type, models_data in results.items():
            json_results[data_type] = {}
            
            for model_name, model_results in models_data.items():
                json_results[data_type][model_name] = {
                    'model_name': model_results['model_name'],
                    'avg_metrics': model_results['avg_metrics'],
                    'classification_report': model_results['classification_report']
                }
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
            
        print(f"Results saved to {json_path}")
        
        # Generate and save visualizations
        self.visualize_results(results, output_dir)
    
    def visualize_results(self, results: Dict[str, Any], output_dir: str) -> None:
        """
        Create visualizations of evaluation results.
        
        Args:
            results: The evaluation results
            output_dir: Directory to save visualizations
        """
        # Set style
        plt.style.use('ggplot')
        
        # Generate comparison plots for each data split
        for data_type, models_data in results.items():
            # Prepare data for plotting
            model_names = []
            accuracies = []
            f1_scores = []
            rouge_l_scores = []
            bertscore_f1_scores = []
            
            for model_name, model_results in models_data.items():
                model_names.append(model_results['model_name'])
                accuracies.append(model_results['avg_metrics']['accuracy'])
                f1_scores.append(model_results['avg_metrics']['f1'])
                rouge_l_scores.append(model_results['avg_metrics']['avg_rougeL_f'])
                bertscore_f1_scores.append(model_results['avg_metrics']['avg_bertscore_f1'])
            
            # Plot accuracy and F1 score
            plt.figure(figsize=(10, 6))
            x = np.arange(len(model_names))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(x - width/2, accuracies, width, label='Accuracy')
            ax.bar(x + width/2, f1_scores, width, label='F1 Score')
            
            ax.set_ylabel('Score')
            ax.set_title(f'Model Performance on {data_type.capitalize()} Data')
            ax.set_xticks(x)
            ax.set_xticklabels(model_names)
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{data_type}_accuracy_f1.png"))
            plt.close()
            
            # Plot ROUGE-L and BERTScore
            plt.figure(figsize=(10, 6))
            x = np.arange(len(model_names))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(x - width/2, rouge_l_scores, width, label='ROUGE-L')
            ax.bar(x + width/2, bertscore_f1_scores, width, label='BERTScore F1')
            
            ax.set_ylabel('Score')
            ax.set_title(f'Text Similarity Metrics on {data_type.capitalize()} Data')
            ax.set_xticks(x)
            ax.set_xticklabels(model_names)
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{data_type}_rouge_bertscore.png"))
            plt.close()
            
            # Create a confusion matrix for the most common classes (top 10)
            for model_name, model_results in models_data.items():
                # Get predictions and ground truth
                predictions = model_results['detailed_results']['predictions']
                ground_truth = model_results['detailed_results']['ground_truth']
                
                # Get the top 10 most common subjects
                subject_counts = pd.Series(ground_truth).value_counts().head(10)
                top_subjects = subject_counts.index.tolist()
                
                # Filter data to only include top subjects
                mask = np.isin(ground_truth, top_subjects)
                filtered_predictions = [p for i, p in enumerate(predictions) if mask[i]]
                filtered_ground_truth = [g for i, g in enumerate(ground_truth) if mask[i]]
                
                # Create confusion matrix
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(
                    filtered_ground_truth, 
                    filtered_predictions, 
                    labels=top_subjects
                )
                
                # Plot confusion matrix
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    cm, 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues',
                    xticklabels=top_subjects,
                    yticklabels=top_subjects
                )
                plt.title(f'Confusion Matrix - {model_results["model_name"]} (Top 10 Subjects)')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=45)
                plt.tight_layout()
                
                # Save plot
                plt.savefig(os.path.join(output_dir, f"{data_type}_{model_name}_confusion_matrix.png"))
                plt.close()
        
        print(f"Visualizations saved to {output_dir}")
    

    def print_summary(self, results: Dict[str, Any]) -> None:
        """
        Print a summary of the evaluation results.
        
        Args:
            results: The evaluation results
        """
        print("\n===== EVALUATION SUMMARY =====")
        
        for data_type, models_data in results.items():
            print(f"\n--- {data_type.upper()} DATA ---")
            
            for model_name, model_results in models_data.items():
                print(f"\n{model_results['model_name']}:")
                
                metrics = model_results['avg_metrics']
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  F1 Score: {metrics['f1']:.4f}")
                print(f"  ROUGE-L: {metrics['avg_rougeL_f']:.4f}")
                print(f"  BERTScore F1: {metrics['avg_bertscore_f1']:.4f}")
        
        print("\n=============================")

    def generate_html_report(self, results: Dict[str, Any], output_file: str = "../evaluation_results/evaluation_report.html") -> None:
        """
        Generate a comprehensive HTML report of evaluation results.
        
        Args:
            results: The evaluation results dictionary
            output_file: Path to save the HTML report
        """
        import base64
        from io import BytesIO
        from datetime import datetime
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Start HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Bill Classification Model Evaluation Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    line-height: 1.6;
                    color: #333;
                }}
                h1, h2, h3, h4 {{
                    color: #2c3e50;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .section {{
                    margin-bottom: 30px;
                    padding: 20px;
                    background-color: #f9f9f9;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                }}
                .metrics-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                .metrics-table th, .metrics-table td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                .metrics-table th {{
                    background-color: #f2f2f2;
                    font-weight: bold;
                }}
                .metrics-table tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .metrics-table tr:hover {{
                    background-color: #f1f1f1;
                }}
                .visualization {{
                    margin: 20px 0;
                    text-align: center;
                }}
                .visualization img {{
                    max-width: 100%;
                    height: auto;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                    border-radius: 4px;
                }}
                .sample-predictions {{
                    overflow-x: auto;
                }}
                .model-comparison {{
                    display: flex;
                    justify-content: space-between;
                    flex-wrap: wrap;
                }}
                .model-card {{
                    width: 48%;
                    margin-bottom: 20px;
                    padding: 15px;
                    background-color: #fff;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                }}
                @media (max-width: 768px) {{
                    .model-card {{
                        width: 100%;
                    }}
                }}
                .highlight {{
                    background-color: #e8f4f8;
                    padding: 2px 5px;
                    border-radius: 3px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Bill Classification Model Evaluation Report</h1>
                <p>Generated on: {timestamp}</p>
                
                <div class="section">
                    <h2>Executive Summary</h2>
                    <p>This report compares the performance of the base DistilBERT model and the fine-tuned model
                    on the task of classifying bill subjects based on their features.</p>
        """
        
        # Add executive summary metrics
        html_content += "<h3>Performance Overview</h3><div class='model-comparison'>"
        
        for data_type, models_data in results.items():
            for model_name, model_results in models_data.items():
                metrics = model_results['avg_metrics']
                display_name = model_results['model_name']
                
                html_content += f"""
                <div class="model-card">
                    <h4>{display_name}</h4>
                    <table class="metrics-table">
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Accuracy</td>
                            <td>{metrics['accuracy']:.4f}</td>
                        </tr>
                        <tr>
                            <td>Precision</td>
                            <td>{metrics['precision']:.4f}</td>
                        </tr>
                        <tr>
                            <td>Recall</td>
                            <td>{metrics['recall']:.4f}</td>
                        </tr>
                        <tr>
                            <td>F1 Score</td>
                            <td>{metrics['f1']:.4f}</td>
                        </tr>
                        <tr>
                            <td>ROUGE-L</td>
                            <td>{metrics['avg_rougeL_f']:.4f}</td>
                        </tr>
                        <tr>
                            <td>BERTScore</td>
                            <td>{metrics['avg_bertscore_f1']:.4f}</td>
                        </tr>
                    </table>
                </div>
                """
        
        html_content += "</div>"
        
        # Add visualizations from files
        html_content += """
                </div>
                
                <div class="section">
                    <h2>Performance Visualizations</h2>
        """
        
        # Add visualization images from files
        for data_type in results.keys():
            html_content += f"""
                    <h3>{data_type.capitalize()} Data Results</h3>
                    
                    <div class="visualization">
                        <h4>Accuracy and F1 Score Comparison</h4>
                        <img src="../evaluation_results/{data_type}_accuracy_f1.png" alt="Accuracy and F1 Score Comparison">
                    </div>
                    
                    <div class="visualization">
                        <h4>ROUGE-L and BERTScore Comparison</h4>
                        <img src="../evaluation_results/{data_type}_rouge_bertscore.png" alt="ROUGE-L and BERTScore Comparison">
                    </div>
                    
                    <h4>Confusion Matrices</h4>
                    <div class="model-comparison">
            """
            
            for model_name in results[data_type].keys():
                html_content += f"""
                        <div class="visualization">
                            <img src="../evaluation_results/{data_type}_{model_name}_confusion_matrix.png" alt="Confusion Matrix">
                            <p>{results[data_type][model_name]['model_name']} - Top 10 Subjects</p>
                        </div>
                """
            
            html_content += "</div>"
        
        # Sample predictions
        html_content += """
                </div>
                
                <div class="section">
                    <h2>Sample Predictions</h2>
                    <div class="sample-predictions">
        """
        
        # Add sample predictions for each model
        for data_type, models_data in results.items():
            html_content += f"<h3>{data_type.capitalize()} Data - Sample Predictions</h3>"
            
            for model_name, model_results in models_data.items():
                # Get sample predictions (first 10)
                predictions = model_results['detailed_results']['predictions'][:10]
                ground_truth = model_results['detailed_results']['ground_truth'][:10]
                input_texts = model_results['detailed_results']['input_texts'][:10]
                rouge_l = model_results['detailed_results']['rougeL_f'][:10]
                bertscore = model_results['detailed_results']['bertscore_f1'][:10]
                
                html_content += f"""
                    <h4>{model_results['model_name']}</h4>
                    <table class="metrics-table">
                        <tr>
                            <th>Input Text (truncated)</th>
                            <th>Ground Truth</th>
                            <th>Prediction</th>
                            <th>ROUGE-L</th>
                            <th>BERTScore</th>
                        </tr>
                """
                
                for i in range(len(predictions)):
                    # Truncate input text for display
                    truncated_input = input_texts[i][:200] + "..." if len(input_texts[i]) > 200 else input_texts[i]
                    
                    # Add color class based on correctness
                    row_class = "correct" if predictions[i] == ground_truth[i] else ""
                    
                    html_content += f"""
                        <tr class="{row_class}">
                            <td>{truncated_input}</td>
                            <td>{ground_truth[i]}</td>
                            <td>{predictions[i]}</td>
                            <td>{rouge_l[i]:.4f}</td>
                            <td>{bertscore[i]:.4f}</td>
                        </tr>
                    """
                
                html_content += "</table>"
        
        # Detailed metrics
        html_content += """
                    </div>
                </div>
                
                <div class="section">
                    <h2>Detailed Metrics</h2>
        """
        
        for data_type, models_data in results.items():
            html_content += f"<h3>{data_type.capitalize()} Data - Model Comparison</h3>"
            
            # Create a table to compare all metrics side by side
            html_content += """
                    <table class="metrics-table">
                        <tr>
                            <th>Metric</th>
            """
            
            # Add model names as columns
            for model_name, model_results in models_data.items():
                html_content += f"<th>{model_results['model_name']}</th>"
            
            html_content += "</tr>"
            
            # Add metrics as rows
            metric_names = [
                "Accuracy", "Precision", "Recall", "F1 Score", 
                "ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "BERTScore"
            ]
            
            metric_keys = [
                "accuracy", "precision", "recall", "f1",
                "avg_rouge1_f", "avg_rouge2_f", "avg_rougeL_f", "avg_bleu", "avg_bertscore_f1"
            ]
            
            for i, metric in enumerate(metric_names):
                html_content += f"""
                    <tr>
                        <td>{metric}</td>
                """
                
                for model_name, model_results in models_data.items():
                    html_content += f"<td>{model_results['avg_metrics'][metric_keys[i]]:.4f}</td>"
                
                html_content += "</tr>"
            
            html_content += "</table>"
            
            # Add per-class metrics
            html_content += "<h3>Per-Class Performance</h3>"
            
            for model_name, model_results in models_data.items():
                html_content += f"""
                    <h4>{model_results['model_name']}</h4>
                    <div class="sample-predictions">
                        <table class="metrics-table">
                            <tr>
                                <th>Class</th>
                                <th>Precision</th>
                                <th>Recall</th>
                                <th>F1 Score</th>
                                <th>Support</th>
                            </tr>
                """
                
                # Add per-class metrics from classification report
                for class_name, metrics in model_results['classification_report'].items():
                    # Skip avg metrics
                    if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                        continue
                    
                    html_content += f"""
                            <tr>
                                <td>{class_name}</td>
                                <td>{metrics['precision']:.4f}</td>
                                <td>{metrics['recall']:.4f}</td>
                                <td>{metrics['f1-score']:.4f}</td>
                                <td>{metrics['support']}</td>
                            </tr>
                    """
                
                html_content += """
                        </table>
                    </div>
                """
        
        # Closing tags
        html_content += """
                </div>
            </div>
        </body>
        </html>
        """
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"HTML report generated at {output_file}")


def main():
    """
    Main function to test the evaluation pipeline.
    """
    print("Testing Bill Classifier Evaluator...")
    
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
 


if __name__ == "__main__":
    main()