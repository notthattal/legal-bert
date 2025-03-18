# legal-bert




## Folder Structure
```
legal-bert/
├── data/                           # Dataset files
│   ├── label_mapping.json          # Mapping of labels for classification
│   ├── masterCategorized.csv       # Categorized dataset
│   ├── test.csv                    # Test dataset
├── evaluation_results/             # Evaluation metrics and model results
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
├── venv/                           # Virtual environment (not typically included in version control)
│   └── .env
├── .gitignore                      # Git ignore file
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
└── run_legal_bert.ipynb            # Jupyter notebook for running and experimenting with the model
```
