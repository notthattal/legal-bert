# Legal-BERT: Enhancing DistilBERT for Predicting Legislative Bill Subjects from Metadata

---

## Project Overview

This project aims to enhance DistilBERT's capabilities in predicting the subject category of legislative bills based solely on their metadata, specifically the bill's title and related factual legal knowledge. 

By incorporating factual knowledge about law and legislation into the model's training, we seek to improve its predictive performance and practical applicability in legislative analysis tasks.

---

## Choice of Focal Property

The focal property selected for this project is **factual legal knowledge**, specifically related to legislative bills. Legislative bills are complex documents with specialized language and terminology. Accurately predicting the subject category of a bill based solely on its metadata (e.g., title) requires a deep understanding of legal terminology, context, and factual knowledge.

We chose this focal property because:

- Incorporating factual legal knowledge helps the model better distinguish subtle differences between categories.
- Improving classification accuracy can significantly aid legislative analysts, researchers, and policymakers.

---

## Chosen Model: DistilBERT ("distilbert-base-uncased")

We selected **DistilBERT (`distilbert-base-uncased`)** as our base model due to its balance between computational efficiency and predictive performance. DistilBERT is a distilled version of BERT that retains most of BERT's capabilities while being faster and lighter.

### Why DistilBERT?

- **Efficiency**: Smaller size and faster inference speed compared to traditional BERT models.
- **Performance**: Proven effectiveness in text classification tasks.
- **Compatibility**: Easy integration with Hugging Face's Transformers library for fine-tuning and deployment.

---

## Data Collection & Preparation

### Dataset Overview

Our dataset consists of legislative bill metadata categorized into predefined subjects. The data was carefully collected from [official government sources.](https://www.congress.gov/bill/)

### Data Files:

- **`masterCategorized.csv`**: Contains labeled bill titles used for both training and evaluation (train-validation split). This data consists of all bills between the 93rd and 118th congress, specifically before November 2024. ***NOTE: this file in the repo is only a subset of the original file used for training, due to size constraints***
- - **`subsetMasterCategorized.csv`**: This file is only a subset of the original file used for training. `masterCategorized.csv` is over 100MB so we uploaded a subset of the data for code testing
- **`test.csv`**: Manually collected independent test set containing bill titles not present in the training data, used exclusively for final evaluation. This data consists of 5 bills per category from the 119th congress. In total, this test set contains 155 bills
- **`label_mapping.json`**: Maps numerical labels to the Subject (Policy Area) names extracted from the .gov website

### Data Collection Process:

1. **Initial Data Gathering**: Collected publicly available legislative bill titles and their corresponding subject categories and metadata from [official government sources.](https://www.congress.gov/bill/), specifically from the 93rd to the 118th congress for the train set
2. **Data Cleaning & Preprocessing**: Standardized text formatting, removed duplicates, and ensured consistent labeling across categories using scripts provided in `process_data.py`. Given that some bills' subjects were missed during data collection, we utilized gpt-4-turbo to fill out those subjects given the label mapping we created 

### Data Usage:

- The labeled data (`masterCategorized.csv`) was split into training (80%) and validation (20%) subsets during fine-tuning
- An independent test set (`test.csv`) was reserved exclusively for evaluation purposes after model training

---

## Fine-Tuning Approach

Our fine-tuning approach involved adapting DistilBERT specifically for the task of classifying legislative bill subjects based on their titles and metadata:

1. **Tokenization**:
   - Utilized DistilBERT's tokenizer (`distilbert-base-uncased`) to tokenize bill titles into appropriate input format for the model.

2. **Fine-Tuning Procedure**:
   - Loaded pre-trained DistilBERT weights from Hugging Face Transformers.
   - Added a classification head layer tailored specifically for our number of subject categories.
   - Fine-tuned the entire model end-to-end using our labeled dataset (`masterCategorized.csv`) with cross-entropy loss.
   - Hyperparameters were tuned through validation set performance monitoring
   - Monitoring was done using the wandb library

3. **Implementation Details**:
   - Fine-tuning scripts are provided in `train.py` and called in `legal_bert.py`
   - Training was conducted using Google Colab via the provided notebook (`run_legal_bert.ipynb`) for GPU acceleration, using A100. The notebook utilizes the main function `legal_bert.py`

---

## Evaluation Strategy

### Evaluation Metrics:

To comprehensively evaluate our model's performance, we employed standard classification metrics:

- **Accuracy**: Overall percentage of correctly predicted samples.
- **Precision, Recall, F1-score (Macro-average)**: To account for class imbalance and provide insights into class-wise performance.
- **Confusion Matrix**: To visualize misclassifications between different subject categories clearly
- **ROUGE and BLEU Scores**: For comparing generated text to reference text, where the reference text represents the bill's subject collected from the .gov website 
- **BERT score**: For semantic similarity between texts. We realized that BERT score on its own would not be enough as there are cases where the BERT score is very high, but the prediction is way off. Combined with other metrics, BERT score provides a more comprehensive overview of the model's performance

### Evaluation Datasets:

- **Validation Set**: Derived from `masterCategorized.csv`, used during fine-tuning to optimize hyperparameters and prevent overfitting.
- **Independent Test Set (`test.csv`)**: Manually collected dataset containing unseen bills used exclusively for final evaluation to assess generalization capability.

### Evaluation Implementation:

Evaluation scripts provided in `evaluate.py` include functionality to calculate metrics, generate confusion matrices, plots, and HTML reports summarizing results clearly. Results are stored systematically within the `evaluation_results/` directory.

**To view our evaluation results, please open the HTML file in your browser titled `evaluation_report.html`**

---

## Project Structure

```
legal-bert/
├── data/                           # Dataset files
│   ├── label_mapping.json          # Mapping of labels for classification
│   ├── masterCategorized.csv       # Categorized dataset (used for training and used for evaluation too since we are evaluating on the train set as well)
│   ├── test.csv                    # Test dataset (includes bills not in the training set)
├── evaluation_results/             # Evaluation metrics and model results, includes HTML, plots, and confusion matrices
├── models/                         # Pretrained and fine-tuned model files
│   └── fine_tuned_legalbert/
│       ├── config.json
│       ├── model.safetensors       # Fine-tuned model weights
│       ├── special_tokens_map.json
│       ├── tokenizer_config.json
│       ├── vocab.txt
├── src/                            # Source code for model training and evaluation
│   ├── __init__.py
│   ├── evaluate.py
│   ├── legal_bert.py               # Core model implementation
│   ├── process_data.py             # Data processing utilities
│   ├── train.py                    # Training script
├── venv/                           
├── .gitignore                      # Git ignore file
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
└── run_legal_bert.ipynb            # Jupyter notebook used for training the model in Colab
```

---

## Getting Started

### Installation & Setup:

1. Clone this repository:
```bash
git clone https://github.com/yourusername/legal-bert.git
cd legal-bert/
```

2. Create a virtual environment & install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # For Windows use venv\Scripts\activate.bat
pip install -r requirements.txt
```

---

## Conclusion

This project demonstrates how effectively leveraging factual legal knowledge can significantly enhance DistilBERT's predictive capabilities in classifying legislative bills based on metadata alone. The carefully designed evaluation strategy ensures robust assessment of our model's performance, emphasizing both accuracy and generalization capabilities.

---
