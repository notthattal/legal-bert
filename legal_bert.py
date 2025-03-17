from captum.attr import LayerIntegratedGradients
import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
import re
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import wandb

device = torch.device('mps') if torch.backends.mps.is_available() else (
    torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
)
print(f"Using device: {device}")

class EthicalAlignmentFramework:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        self.category_weights = {
            'Civil Rights and Liberties': 1.2,
            'Immigration': 1.0,
            'Economics': 1.0,
        }

        self.sensitive_terms = {
            'rights': 1.3,
            'discrimination': 1.3,
            'equality': 1.3,
            'minority': 1.2,
            'gender': 1.2,
            'race': 1.2,
            'religion': 1.2,
            'disability': 1.2,
            'refugee': 1.1,
            'immigrant': 1.1,
        }

    def ethically_adjusted_prediction(self, text: str):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            base_probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        adjusted_probs = self.apply_ethical_weights(text, base_probs[0])

        predicted_class = torch.argmax(adjusted_probs).item()

        return {
            'text': text,
            'original_prediction': torch.argmax(base_probs).item(),
            'adjusted_prediction': predicted_class,
            'original_confidence': float(torch.max(base_probs).item()),
            'adjusted_confidence': float(adjusted_probs[predicted_class]),
            'ethical_adjustment_factors': self.calculate_adjustment_factors(text)
        }

    def apply_ethical_weights(self, text: str, base_probs: torch.Tensor):
        adjustment_factors = self.calculate_adjustment_factors(text)
        adjusted_probs = base_probs.clone()

        for category, weight in self.category_weights.items():
            if category == 'Civil Rights and Liberties':
                label_index = 14
            elif category == 'Immigration':
                label_index = 76
            elif category == 'Economics':
                label_index = 41

            adjusted_probs[label_index] *= weight * adjustment_factors.get('term_weight', 1.0)

        adjusted_probs = adjusted_probs / adjusted_probs.sum()

        return adjusted_probs

    def calculate_adjustment_factors(self, text: str):
        factors = {'term_weight': 1.0}
        term_weights = []

        text = text.lower()

        for term, weight in self.sensitive_terms.items():
            if term in text:
                term_weights.append(weight)

        if term_weights:
            factors['term_weight'] = np.mean(term_weights)

        return factors

    def evaluate_ethical_alignment(self, texts):
        results = []

        for text in texts:
            prediction = self.ethically_adjusted_prediction(text)
            results.append(prediction)

        return pd.DataFrame(results)

class LegislativeBiasAnalyzer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        self.sensitive_terms = {
            'racial': ['race', 'racial', 'ethnic', 'minority', 'discrimination', 'african', 'asian', 'hispanic'],
            'gender': ['gender', 'woman', 'women', 'man', 'men', 'female', 'male', 'sex'],
            'religious': ['religion', 'religious', 'faith', 'belief', 'worship', 'church', 'mosque', 'temple'],
            'nationality': ['citizen', 'immigrant', 'foreign', 'national', 'alien', 'refugee', 'asylum'],
            'socioeconomic': ['poor', 'poverty', 'income', 'wealth', 'economic', 'class', 'welfare']
        }

    def analyze_bias_in_text(self, text: str):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        term_analysis = self._analyze_sensitive_terms(tokens)

        return {
            'text': text,
            'prediction': torch.argmax(probs).item(),
            'confidence': float(torch.max(probs).item()),
            'sensitive_terms': term_analysis
        }

    def _analyze_sensitive_terms(self, tokens):
        results = {}

        for category, terms in self.sensitive_terms.items():
            found_terms = []

            for term in terms:
                term_tokens = self.tokenizer.tokenize(term)
                if any(t in tokens for t in term_tokens):
                    found_terms.append(term)

            if found_terms:
                results[category] = found_terms

        return results

    def analyze_category_bias(self, texts_by_category):
        results = []

        for category, texts in texts_by_category.items():
            for text in texts:
                analysis = self.analyze_bias_in_text(text)
                analysis['intended_category'] = category
                results.append(analysis)

        return pd.DataFrame(results)

    def visualize_bias_patterns(self, df: pd.DataFrame):
        plt.figure(figsize=(15, 6))
        sns.boxplot(data=df, x='intended_category', y='confidence')
        plt.xticks(rotation=45, ha='right')
        plt.title('Prediction Confidence by Category')
        plt.tight_layout()
        plt.show()

        sensitive_counts = {category: [] for category in self.sensitive_terms.keys()}
        for _, row in df.iterrows():
            for category in self.sensitive_terms.keys():
                if category in row['sensitive_terms']:
                    sensitive_counts[category].append(len(row['sensitive_terms'][category]))
                else:
                    sensitive_counts[category].append(0)

        plt.figure(figsize=(10, 6))
        plt.bar(sensitive_counts.keys(), [np.mean(counts) for counts in sensitive_counts.values()])
        plt.xticks(rotation=45, ha='right')
        plt.title('Average Sensitive Term Usage by Category')
        plt.tight_layout()
        plt.show()

class ExtendedBiasAnalyzer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def analyze_intersectional_bias(self, texts_by_category):
        results = []

        for category, texts in texts_by_category.items():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

                intersectional_terms = self.check_intersectional_terms(text)

                results.append({
                    'category': category,
                    'text': text,
                    'confidence': float(torch.max(probs).item()),
                    'intersectional_terms': intersectional_terms,
                    'prediction': torch.argmax(probs).item()
                })

        return pd.DataFrame(results)

    def check_intersectional_terms(self, text: str):
        intersectional_categories = {
            'race_gender': ['african american women', 'latina women', 'asian men'],
            'religion_nationality': ['muslim immigrants', 'jewish refugees'],
            'socioeconomic_race': ['minority poverty', 'racial wealth gap'],
            'gender_economics': ['wage gap', 'working mothers'],
            'disability_employment': ['disabled workers', 'workplace accommodation']
        }

        found_terms = {}
        for category, terms in intersectional_categories.items():
            matches = [term for term in terms if term.lower() in text.lower()]
            if matches:
                found_terms[category] = matches

        return found_terms

    def calculate_statistical_significance(self, df: pd.DataFrame):
        categories = df['category'].unique()

        print("\nStatistical Analysis of Confidence Differences:")
        print("-" * 80)

        category_groups = [group['confidence'].values for name, group in df.groupby('category')]
        f_stat, p_value = stats.f_oneway(*category_groups)

        print(f"One-way ANOVA results:")
        print(f"F-statistic: {f_stat:.4f}")
        print(f"p-value: {p_value:.4f}")

        print("\nPairwise t-tests:")
        for i in range(len(categories)):
            for j in range(i+1, len(categories)):
                cat1_data = df[df['category'] == categories[i]]['confidence']
                cat2_data = df[df['category'] == categories[j]]['confidence']
                t_stat, p_val = stats.ttest_ind(cat1_data, cat2_data)
                print(f"{categories[i]} vs {categories[j]}:")
                print(f"t-statistic: {t_stat:.4f}")
                print(f"p-value: {p_val:.4f}")
                print()

    def visualize_extended_analysis(self, df: pd.DataFrame):
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='category', y='confidence', ci=95)
        plt.title('Prediction Confidence by Category with 95% Confidence Intervals')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

class LegislativeDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
class LegislativeBillAnalyzer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        self.label_mapping = {
            0: "Agriculture and Food",
            5: "Armed Forces and National Security",
            14: "Civil Rights and Liberties, Minority Issues",
            41: "Commerce",
            44: "Crime and Law Enforcement",
            56: "Education",
            58: "Energy",
            59: "Environmental Protection",
            61: "Finance and Financial Sector",
            70: "Foreign Trade and International Finance",
            71: "Government Operations and Politics",
            72: "Health",
            75: "Housing and Community Development",
            76: "Immigration",
            77: "Infrastructure",
            79: "Intellectual Property",
            80: "International Affairs",
            85: "Labor and Employment",
            86: "Law",
            93: "Native Americans",
            98: "Public Debt",
            101: "Public Lands and Natural Resources",
            105: "Science, Technology, Communications",
            107: "Social Welfare",
            110: "Taxation",
            113: "Transportation and Public Works",
            114: "Veterans",
            115: "Water Resources Development"
        }

        self.ref_token_id = self.tokenizer.pad_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.cls_token_id = self.tokenizer.cls_token_id

        self.lig = LayerIntegratedGradients(
            self.forward_func,
            self.model.bert.embeddings
        )

    def get_category_name(self, label_id: int) -> str:
        return self.label_mapping.get(label_id, f"Unknown Category (Label_{label_id})")

    def forward_func(self, inputs: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        return self.model(inputs, attention_mask=attention_mask).logits

    def analyze_bill(self, text: str):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probs).item()

        attributions = self.get_integrated_gradients(text, predicted_class)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        top_5_indices = torch.topk(probs[0], 5).indices.cpu().numpy()
        top_5_probs = torch.topk(probs[0], 5).values.cpu().numpy()
        top_5_predictions = [
            (self.get_category_name(idx), float(prob))
            for idx, prob in zip(top_5_indices, top_5_probs)
        ]

        return {
            'text': text,
            'predicted_label_id': predicted_class,
            'predicted_category': self.get_category_name(predicted_class),
            'confidence': float(probs[0][predicted_class]),
            'tokens': tokens,
            'attributions': attributions,
            'top_5_predictions': top_5_predictions
        }

    def get_integrated_gradients(self, text: str, target_class: int):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        ref_input_ids = torch.tensor(
            [[self.cls_token_id] + [self.ref_token_id] * (inputs['input_ids'].shape[1] - 2) + [self.sep_token_id]],
            device=device
        )

        attributions, delta = self.lig.attribute(
            inputs=inputs['input_ids'],
            baselines=ref_input_ids,
            target=target_class,
            additional_forward_args=(inputs['attention_mask'],),
            return_convergence_delta=True
        )

        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        return attributions.cpu().detach().numpy()

    def visualize_attributions(self, tokens, attributions: np.ndarray, title: str = "Token Attributions"):
        plt.figure(figsize=(15, 5))
        plt.bar(range(len(tokens)), attributions)
        plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
        plt.title(title)
        plt.xlabel("Tokens")
        plt.ylabel("Attribution Score")
        plt.tight_layout()
        plt.show()

def get_congress_cats():
    return [
        'Agriculture and Food',
        'Armed Forces and National Security',
        'Civil Rights and Liberties, Minority Issues',
        'Commerce',
        'Crime and Law Enforcement',
        'Economics and Public Finance',
        'Education',
        'Energy',
        'Environmental Protection',
        'Families',
        'Finance and Financial Sector',
        'Foreign Trade and International Finance',
        'Government Operations and Politics',
        'Health',
        'Housing and Community Development',
        'Immigration',
        'International Affairs',
        'Labor and Employment',
        'Law',
        'Native Americans',
        'Public Lands and Natural Resources',
        'Science, Technology, Communications',
        'Social Welfare',
        'Taxation',
        'Transportation and Public Works',
        'Water Resources Development',
        'Infrastructure',
        'Veterans',
        'Public Debt',
        'Intellectual Property',
        'Military Logistics',
        'Telecommunications and Information'
    ]

def get_cat_from_gpt(client, title, categories):
    prompt = f"""
    You are tasked with categorizing legislative bills into the following categories:
    {', '.join(categories)}.
    Based on the title of the bill, assign it to the most appropriate category.
    If it's unclear, assign it to 'other'. Be sure to only return the category and nothing else.

    Title: "{title}"
    """

    chat_completion = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an advanced AI assistant who categorizes legislative bills using knowledge of the U.S Law."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        temperature=0
    )

    category = chat_completion.choices[0].message.content.strip()
    return category
    
# apply each abbreviation replacement to the 'Title' column
def replace_abbreviations(text):
    abbreviation_replacements = {
        r'&c.': 'Etcetera',
        r'&': 'and',
        r'\bet al\b': 'and others',
        r'\bviz\b\.': 'namely',
        r'\bi\.e\b\.': 'that is',
        r'\be\.g\b\.': 'for example'
    }

    for pattern, replacement in abbreviation_replacements.items():
        text = re.sub(pattern, replacement, text)
    return text

def clean_master_df(df):
    df = df.dropna(subset=['Title'])
    df['clean_title'] = df['Title'].apply(replace_abbreviations)
    new_titles = df[df['Title'] != df['clean_title']][['Title', 'Title_cleaned']]

    return df

def categorize_missing_cats(df, client, categories, output_file):
    if 'Subject' not in df.columns:
        df['Subject'] = pd.NA
    
    missing_cats = df[df['Subject'].isna()]

    batch_size = 500
    if len(missing_cats) > 0:
        rows_to_process = missing_cats.head(batch_size)
        df.loc[rows_to_process.index, 'Subject'] = rows_to_process['Title'].apply(
            lambda title: get_cat_from_gpt(client, title, categories)
        )
        df.to_csv(output_file, index=False)

        print(f"Successfully categorized {len(rows_to_process)} rows. Data saved to a CSV.")
    else:
        print("No further categorization necessary")

    return df 

def tokenize_function(texts, tokenizer):
    return tokenizer(texts, padding=True, truncation=True, max_length=512)

def encode_labels(df):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(df)
    num_labels = len(label_encoder.classes_)

    return encoded_labels, num_labels

def get_train_val_datasets(tokenizer, titles, encoded_labels):
    train_texts, val_texts, train_labels, val_labels = train_test_split(titles, encoded_labels, test_size=0.2, random_state=42)

    train_encodings = tokenize_function(train_texts, tokenizer)
    val_encodings = tokenize_function(val_texts, tokenizer)

    train_dataset = LegislativeDataset(train_encodings, train_labels)
    val_dataset = LegislativeDataset(val_encodings, val_labels)
    
    return train_dataset, val_dataset

def train_model(model, tokenizer, train_dataset, val_dataset):
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
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

def evaluate_model(trainer, comprehensive=False):
    eval_results = trainer.evaluate()
    wandb.log({"final_evaluation": eval_results})
    print(f"Evaluation results: {eval_results}")

def get_example_bills():
    return [
        "To amend the Food Security Act to strengthen agricultural programs.",
        "To improve healthcare services and benefits for veterans.",
        "To strengthen enforcement of civil rights laws and improve access to justice.",
        "To develop transportation infrastructure and public transit systems.",
        "To protect endangered species and preserve natural habitats."
    ]

def get_example_ethics_statements():
    return [
        "To strengthen civil rights protections and prevent discrimination based on race, gender, or religion.",
        "To ensure equal access to education and employment opportunities for minority communities.",
        "To reform immigration procedures and establish fair processing of asylum claims.",
        "To protect the rights of immigrant workers and prevent workplace exploitation.",
        "To regulate financial institutions and prevent predatory lending practices.",
        "To promote economic development in underserved communities."
    ]

def get_example_biases(extended=False):
    if extended:
        return {
            'Civil Rights': [
                "To protect against discrimination based on race and gender in employment.",
                "To ensure religious freedom and prevent discrimination in public accommodations.",
                "To address racial disparities in access to education and healthcare."
            ],
            'Economics': [
                "To address economic inequality and promote fair lending practices.",
                "To regulate financial institutions and prevent predatory lending.",
                "To promote economic opportunity in underserved communities."
            ],
            'Immigration': [
                "To establish procedures for processing asylum claims and refugee admissions.",
                "To protect the rights of immigrant workers and prevent exploitation.",
                "To reform immigration policies while ensuring border security."
            ]
        }

    return {
        'Civil Rights': [
            "To strengthen enforcement of civil rights laws and prevent discrimination based on race, color, or national origin.",
            "To ensure equal protection under law regardless of gender or religious beliefs.",
            "To promote fair housing practices and prevent discrimination in housing markets."
        ],
        'Economics': [
            "To reform economic policies and regulate financial institutions.",
            "To improve fiscal management and economic stability measures.",
            "To enhance oversight of banking and financial markets."
        ],
        'Immigration': [
            "To reform immigration policies and strengthen border security measures.",
            "To establish procedures for processing refugee and asylum claims.",
            "To modify requirements for citizenship and naturalization."
        ]
    }

def analyze_legislative_bills(analyzer, texts=None):
    if texts is None:
        texts = get_example_bills()

    for text in texts:
        print("\nAnalyzing:", text)
        results = analyzer.analyze_bill(text)

        print(f"Predicted Category: {results['predicted_category']}")
        print(f"Confidence: {results['confidence']:.4f}")

        print("\nTop 5 predictions:")
        for category, prob in results['top_5_predictions']:
            print(f"{category}: {prob:.4f}")

        analyzer.visualize_attributions(
            results['tokens'],
            results['attributions'],
            f"Token Attributions for {results['predicted_category']}"
        )

        token_importance = list(zip(results['tokens'], results['attributions']))
        token_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        print("\nMost influential tokens:")
        for token, importance in token_importance[:5]:
            print(f"{token}: {importance:.4f}")
        print("-" * 80)

def analyze_legislative_biases(analyzer, texts=None):
    if texts is None:
        texts = get_example_biases()

    results_df = analyzer.analyze_category_bias(texts)

    print("\nBias Analysis Summary:")
    print("-" * 80)

    print("\nPrediction Patterns:")
    for category in texts.keys():
        category_data = results_df[results_df['intended_category'] == category]
        print(f"\n{category}:")
        print(f"Average confidence: {category_data['confidence'].mean():.4f}")
        print("Common sensitive terms:", end=" ")
        all_terms = []
        for terms in category_data['sensitive_terms']:
            all_terms.extend([term for terms_list in terms.values() for term in terms_list])
        if all_terms:
            print(", ".join(set(all_terms)))
        else:
            print("None found")

    analyzer.visualize_bias_patterns(results_df)

def analyze_extended_biases(analyzer, texts=None):
    if texts is None:
        texts = get_example_biases(extended=True)

    results_df = analyzer.analyze_intersectional_bias(texts)
    analyzer.calculate_statistical_significance(results_df)
    analyzer.visualize_extended_analysis(results_df)

def evaluate_model_ethics(aligner, texts=None):
    if texts is None:
        texts = get_example_ethics_statements()

    results = aligner.evaluate_ethical_alignment(texts)

    print("\nEthical Alignment Results:")
    print("-" * 80)

    for _, row in results.iterrows():
        print(f"\nText: {row['text']}")
        print(f"Original Confidence: {row['original_confidence']:.4f}")
        print(f"Adjusted Confidence: {row['adjusted_confidence']:.4f}")
        print(f"Adjustment Factors: {row['ethical_adjustment_factors']}")
        print("-" * 40)

def comprehensive_model_analysis(model, tokenizer):
    legislative_bill_analyzer = LegislativeBillAnalyzer(model, tokenizer)
    analyze_legislative_bills(legislative_bill_analyzer)

    legislative_bias_analyzer = LegislativeBiasAnalyzer(model, tokenizer)
    analyze_legislative_biases(legislative_bias_analyzer)

    extended_bias_analzyer = ExtendedBiasAnalyzer(model, tokenizer)
    analyze_extended_biases(extended_bias_analzyer)

    ethics_aligner = EthicalAlignmentFramework(model, tokenizer)
    evaluate_model_ethics(ethics_aligner)

def main():
    wandb.init(project="legal-bert-classification", name="training-run-1")

    master_file = './data/master.csv'
    master_df = pd.read_csv(master_file)
    
    '''
    Unnecessary since we already have the categorized df

    client = openai.OpenAI(api_key="use gpt api (this process takes ~20 hours)")
    categories_file = './data/masterCategorized.csv'

    master_df = clean_master_df(master_df)
    master_df = pd.read_csv(categories_file)
    master_df = categorize_missing_cats(master_df, client, categories, categories_file)'
    '''
    titles = master_df['Title'].astype(str).tolist()
    categories = master_df['Subject'].tolist()

    encoded_labels, num_labels = encode_labels(categories)

    model_name = "nlpaueb/legal-bert-base-uncased"
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    train_dataset, val_dataset = get_train_val_datasets(tokenizer, titles, encoded_labels)

    trainer = train_model(model, tokenizer, train_dataset, val_dataset)

    model.eval()
    evaluate_model(trainer)
    comprehensive_model_analysis(model, tokenizer)

    wandb.finish()

if __name__=='__main__':
    main()