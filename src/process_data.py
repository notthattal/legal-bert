'''
This script implements DataProcessor, the data processing pipeline.
Given, the data collected from https://www.congress.gov/bill/119th-congress/, this script processed the csv
by cleaning the title of each bill, reducing abbreviations, categorizes bills with missing categories using chat gpt 
and the list of categories manually collected from the website (found under self.categories) and saves
the final csv used for training with the bill title and bill's metadata alongside the bill's subject (or category)
'''
import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv
import re


load_dotenv()

class DataProcessor():
    def __init__(self, data_dir, master_uncategorized_file_name, master_categorized_file_name):
        self.data_dir = data_dir
        self.master_uncategorized_file_path = os.path.join(self.data_dir, master_uncategorized_file_name)
        self.master_categorized_file_path = os.path.join(self.data_dir, master_categorized_file_name)
        self.client = OpenAI(api_key="OPENAI_API_KEY")
         # categories from congress
        self.categories = ['Agriculture and Food','Armed Forces and National Security','Civil Rights and Liberties, Minority Issues',
                           'Commerce','Crime and Law Enforcement','Economics and Public Finance','Education',
                           'Energy','Environmental Protection','Families','Finance and Financial Sector',
                           'Foreign Trade and International Finance','Government Operations and Politics','Health',
                           'Housing and Community Development','Immigration','International Affairs','Labor and Employment',
                           'Law','Native Americans','Public Lands and Natural Resources','Science, Technology, Communications',
                           'Social Welfare','Taxation','Transportation and Public Works','Water Resources Development',
                           'Infrastructure','Veterans','Public Debt','Intellectual Property','Military Logistics',
                           'Telecommunications and Information']

    def read_uncategorized_master_csv(self):
        '''
        Function to read the csv with bills' metadata, but not subjects
        '''
        master_df = pd.read_csv(self.master_uncategorized_file_path)
        return master_df
    
    def clean_master_df(self, df):
        '''
        Function to clean the bills' titles by applying the replace_abbreviations functions
        '''
        df = df.dropna(subset=['Title'])
        df['clean_title'] = df['Title'].apply(self.replace_abbreviations)
        return df
    
    def get_titles(self, df):
        titles = df['Title'].astype(str).tolist()
        return titles 
    
    def get_categories(self, df):
        categories = df['Subject'].tolist()
        return categories
    
    # apply each abbreviation replacement to the 'Title' column
    def replace_abbreviations(self, text):
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
    
    
    def get_cat_from_gpt(self, client, title, categories):
        '''
        Helper function that uses the OpenAI client and gpt-4-turbo to add the bill's missing category
        given the list of categories collected from the congress website. This function is only called if
        categories are not included (i.e missed during data collection)
        '''
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
    
    def categorize_missing_cats(self, df, client, categories, output_file):
        '''
        This function categorizes missing bills' categories by applying the function 
        get_cat_from_gpt (see above). It processes the bills in batches and then saves 
        the result into a csv.
        '''
        if 'Subject' not in df.columns:
            df['Subject'] = pd.NA
        
        missing_cats = df[df['Subject'].isna()]

        batch_size = 500
        if len(missing_cats) > 0:
            rows_to_process = missing_cats.head(batch_size)
            df.loc[rows_to_process.index, 'Subject'] = rows_to_process['Title'].apply(
                lambda title: self.get_cat_from_gpt(client, title, categories)
            )
            df.to_csv(output_file, index=False)

            print(f"Successfully categorized {len(rows_to_process)} rows. Data saved to a CSV.")
        else:
            print("No further categorization necessary")

        return df 
    
    def process_data(self):
        uncategorized_master_df = self.read_uncategorized_master_csv()
        uncategorized_master_df = self.clean_master_df(uncategorized_master_df)
        categorized_master_df = self.categorize_missing_cats(uncategorized_master_df, self.client, self.categories, self.master_categorized_file_path)

        return categorized_master_df
    
def main():
    data_processor = DataProcessor('../data', 'masterUncategorized.csv', 'masterCategorized.csv')
    master_df = data_processor.process_data()
    print(master_df.head(15))


if __name__=='__main__':
    main()