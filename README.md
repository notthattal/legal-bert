# legal-bert




## Folder Structure
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
