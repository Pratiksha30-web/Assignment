"""
Data Cleaning and Preprocessing Module
Handles data cleaning, sentiment labeling, and text preprocessing for reviews
"""

import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')


class DataCleaner:
    """Class to handle data cleaning and preprocessing"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Keep some sentiment-related words
        self.stop_words -= {'not', 'no', 'never', 'neither', 'nor', 'very', 'too', 'good', 'bad'}
        
    def load_data(self, iphone_path, nike_path):
        """Load both datasets"""
        print("Loading datasets...")
        
        # Load iPhone reviews
        iphone_df = pd.read_csv(iphone_path)
        iphone_df['product'] = 'iPhone 14'
        iphone_df['source'] = 'Flipkart'
        
        # Load Nike reviews
        nike_df = pd.read_csv(nike_path)
        nike_df['product'] = 'Nike Shoes'
        nike_df['source'] = 'Nike.com'
        
        # Rename columns to standardize
        iphone_df = iphone_df.rename(columns={
            'title': 'review_title',
            'rating': 'star_rating',
            'review': 'review_text',
            'customer_name': 'reviewer',
            'dates': 'date',
            'customer_location': 'location'
        })
        
        nike_df = nike_df.rename(columns={
            'Reviewer': 'reviewer',
            'Title': 'review_title',
            'Review': 'review_text',
            'Star_rating': 'star_rating',
            'Rating': 'rating_text',
            'Date': 'date',
            'Location': 'location'
        })
        
        print(f"iPhone reviews loaded: {len(iphone_df)}")
        print(f"Nike reviews loaded: {len(nike_df)}")
        
        return iphone_df, nike_df
    
    def clean_data(self, df):
        """Clean the dataset"""
        print("\nCleaning data...")
        
        # Create a copy
        df_clean = df.copy()
        
        # Handle missing values
        print(f"Missing values before cleaning:\n{df_clean.isnull().sum()}")
        
        # Fill missing review text with empty string
        df_clean['review_text'] = df_clean['review_text'].fillna('')
        
        # Fill missing review title with empty string
        df_clean['review_title'] = df_clean['review_title'].fillna('')
        
        # Drop rows where review_text is empty after filling
        df_clean = df_clean[df_clean['review_text'].str.strip() != '']
        
        # Handle star_rating - convert to numeric
        df_clean['star_rating'] = pd.to_numeric(df_clean['star_rating'], errors='coerce')
        
        # Drop rows with missing ratings
        df_clean = df_clean.dropna(subset=['star_rating'])
        
        # Remove duplicates based on review_text
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=['review_text'], keep='first')
        duplicates_removed = initial_count - len(df_clean)
        print(f"Duplicates removed: {duplicates_removed}")
        
        # Clean review text - remove "READ MORE" and similar artifacts
        df_clean['review_text'] = df_clean['review_text'].str.replace('READ MORE', '', case=False)
        df_clean['review_text'] = df_clean['review_text'].str.strip()
        
        print(f"\nRows after cleaning: {len(df_clean)}")
        
        return df_clean
    
    def create_sentiment_labels(self, df):
        """Create binary sentiment labels based on ratings"""
        print("\nCreating sentiment labels...")
        
        df_labeled = df.copy()
        
        # Create sentiment labels
        # Rating >= 4 -> Positive
        # Rating <= 2 -> Negative
        # Rating == 3 -> Neutral (we'll exclude these or classify separately)
        
        def assign_sentiment(rating):
            if rating >= 4:
                return 'Positive'
            elif rating <= 2:
                return 'Negative'
            else:
                return 'Neutral'
        
        df_labeled['sentiment'] = df_labeled['star_rating'].apply(assign_sentiment)
        
        # For binary classification, we'll exclude neutral
        df_binary = df_labeled[df_labeled['sentiment'] != 'Neutral'].copy()
        
        print(f"Sentiment distribution (all):\n{df_labeled['sentiment'].value_counts()}")
        print(f"\nSentiment distribution (binary - excluding neutral):\n{df_binary['sentiment'].value_counts()}")
        
        # Also create numeric label for modeling
        df_binary['sentiment_label'] = df_binary['sentiment'].map({'Positive': 1, 'Negative': 0})
        
        return df_labeled, df_binary
    
    def preprocess_text(self, text):
        """Preprocess a single text"""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove emails
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags (for social media data)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                lemmatized = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemmatized)
        
        # Join tokens back
        processed_text = ' '.join(processed_tokens)
        
        return processed_text
    
    def process_all_texts(self, df):
        """Process all review texts"""
        print("\nPreprocessing text data...")
        
        df_processed = df.copy()
        
        # Combine review title and text for more context
        df_processed['combined_text'] = (df_processed['review_title'].fillna('') + ' ' + 
                                         df_processed['review_text'].fillna('')).str.strip()
        
        # Preprocess the combined text
        df_processed['processed_text'] = df_processed['combined_text'].apply(self.preprocess_text)
        
        # Remove rows with empty processed text
        df_processed = df_processed[df_processed['processed_text'].str.strip() != '']
        
        # Calculate text length metrics
        df_processed['text_length'] = df_processed['review_text'].str.len()
        df_processed['word_count'] = df_processed['review_text'].str.split().str.len()
        
        print(f"Text preprocessing complete. Final rows: {len(df_processed)}")
        
        return df_processed
    
    def clean_date_column(self, df):
        """Clean and parse date column"""
        df_clean = df.copy()
        
        # Try to parse dates
        try:
            df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
            df_clean['year'] = df_clean['date'].dt.year
            df_clean['month'] = df_clean['date'].dt.month
            df_clean['year_month'] = df_clean['date'].dt.to_period('M')
        except:
            print("Warning: Could not parse dates properly")
            df_clean['year'] = np.nan
            df_clean['month'] = np.nan
            df_clean['year_month'] = np.nan
        
        return df_clean


def main():
    """Main execution function"""
    
    # Initialize cleaner
    cleaner = DataCleaner()
    
    # Load data
    iphone_df, nike_df = cleaner.load_data(
        'iphone14_customer_review.csv',
        'web_scraped.csv'
    )
    
    # Clean both datasets
    iphone_clean = cleaner.clean_data(iphone_df)
    nike_clean = cleaner.clean_data(nike_df)
    
    # Combine datasets
    print("\n" + "="*50)
    print("COMBINING DATASETS")
    print("="*50)
    
    # Select common columns
    common_cols = ['review_title', 'review_text', 'star_rating', 'reviewer', 
                   'date', 'location', 'product', 'source']
    
    iphone_combined = iphone_clean[common_cols].copy()
    nike_combined = nike_clean[common_cols].copy()
    
    # Combine
    combined_df = pd.concat([iphone_combined, nike_combined], ignore_index=True)
    print(f"\nTotal combined reviews: {len(combined_df)}")
    
    # Create sentiment labels
    df_all_sentiment, df_binary_sentiment = cleaner.create_sentiment_labels(combined_df)
    
    # Process text
    df_final = cleaner.process_all_texts(df_binary_sentiment)
    
    # Clean dates
    df_final = cleaner.clean_date_column(df_final)
    
    # Save cleaned datasets
    print("\n" + "="*50)
    print("SAVING CLEANED DATASETS")
    print("="*50)
    
    # Save raw combined data
    combined_df.to_csv('cleaned_data_raw.csv', index=False)
    print("✓ Saved: cleaned_data_raw.csv")
    
    # Save data with all sentiments (including neutral)
    df_all_sentiment.to_csv('cleaned_data_all_sentiments.csv', index=False)
    print("✓ Saved: cleaned_data_all_sentiments.csv")
    
    # Save final processed data (binary sentiment, processed text)
    df_final.to_csv('cleaned_data_processed.csv', index=False)
    print("✓ Saved: cleaned_data_processed.csv")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    print(f"\nFinal dataset shape: {df_final.shape}")
    print(f"\nSentiment distribution:")
    print(df_final['sentiment'].value_counts())
    print(f"\nProduct distribution:")
    print(df_final['product'].value_counts())
    print(f"\nAverage rating by product:")
    print(df_final.groupby('product')['star_rating'].mean())
    print(f"\nAverage word count: {df_final['word_count'].mean():.2f}")
    print(f"Median word count: {df_final['word_count'].median():.2f}")
    
    print("\n✓ Data cleaning and preprocessing complete!")
    
    return df_final


if __name__ == "__main__":
    df = main()

