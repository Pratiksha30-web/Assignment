"""
Sentiment Prediction Model Training Module
Trains multiple ML models and evaluates their performance
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')


class SentimentModelTrainer:
    """Class for training and evaluating sentiment prediction models"""
    
    def __init__(self, df):
        self.df = df
        self.models = {}
        self.vectorizer = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_vec = None
        self.X_test_vec = None
        self.results = []
        
    def prepare_data(self, test_size=0.3, random_state=42):
        """Prepare data for training"""
        print("="*70)
        print("PREPARING DATA FOR MODEL TRAINING")
        print("="*70)
        
        # Get features and labels
        X = self.df['processed_text'].values
        y = self.df['sentiment_label'].values
        
        print(f"\nTotal samples: {len(X)}")
        print(f"Positive samples: {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)")
        print(f"Negative samples: {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nTraining set size: {len(self.X_train)} ({(1-test_size)*100:.0f}%)")
        print(f"Test set size: {len(self.X_test)} ({test_size*100:.0f}%)")
        
        # Vectorize text using TF-IDF
        print("\nVectorizing text using TF-IDF...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # Use both unigrams and bigrams
            min_df=2,
            max_df=0.95
        )
        
        self.X_train_vec = self.vectorizer.fit_transform(self.X_train)
        self.X_test_vec = self.vectorizer.transform(self.X_test)
        
        print(f"Feature matrix shape: {self.X_train_vec.shape}")
        print(f"Number of features: {self.X_train_vec.shape[1]}")
        
    def train_models(self):
        """Train multiple models"""
        print("\n" + "="*70)
        print("TRAINING MODELS")
        print("="*70)
        
        # Define models
        models_to_train = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Naive Bayes': MultinomialNB()
        }
        
        # Train each model
        for name, model in models_to_train.items():
            print(f"\n{'='*70}")
            print(f"Training {name}...")
            print(f"{'='*70}")
            
            # Train
            model.fit(self.X_train_vec, self.y_train)
            
            # Predict
            y_pred_train = model.predict(self.X_train_vec)
            y_pred_test = model.predict(self.X_test_vec)
            
            # Evaluate
            train_accuracy = accuracy_score(self.y_train, y_pred_train)
            test_accuracy = accuracy_score(self.y_test, y_pred_test)
            precision = precision_score(self.y_test, y_pred_test)
            recall = recall_score(self.y_test, y_pred_test)
            f1 = f1_score(self.y_test, y_pred_test)
            
            print(f"\nTraining Accuracy: {train_accuracy:.4f}")
            print(f"Test Accuracy:     {test_accuracy:.4f}")
            print(f"Precision:         {precision:.4f}")
            print(f"Recall:            {recall:.4f}")
            print(f"F1 Score:          {f1:.4f}")
            
            # Confusion matrix
            cm = confusion_matrix(self.y_test, y_pred_test)
            print(f"\nConfusion Matrix:")
            print(f"                Predicted")
            print(f"                Neg  Pos")
            print(f"Actual Neg      {cm[0][0]:4d} {cm[0][1]:4d}")
            print(f"       Pos      {cm[1][0]:4d} {cm[1][1]:4d}")
            
            # Store model and results
            self.models[name] = {
                'model': model,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm,
                'y_pred': y_pred_test
            }
            
            self.results.append({
                'Model': name,
                'Train Accuracy': train_accuracy,
                'Test Accuracy': test_accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1
            })
        
        print("\n" + "="*70)
        print("MODEL TRAINING COMPLETE")
        print("="*70)
        
    def compare_models(self):
        """Compare all models"""
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        
        # Create comparison dataframe
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('F1 Score', ascending=False)
        
        print("\n")
        print(results_df.to_string(index=False))
        
        # Find best model
        best_model_name = results_df.iloc[0]['Model']
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   F1 Score: {results_df.iloc[0]['F1 Score']:.4f}")
        print(f"   Test Accuracy: {results_df.iloc[0]['Test Accuracy']:.4f}")
        
        return best_model_name, results_df
    
    def plot_model_comparison(self, results_df):
        """Plot model comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = ['Test Accuracy', 'Precision', 'Recall', 'F1 Score']
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        
        for idx, (metric, color) in enumerate(zip(metrics, colors)):
            ax = axes[idx // 2, idx % 2]
            
            data = results_df.sort_values(metric, ascending=True)
            bars = ax.barh(data['Model'], data[metric], color=color, edgecolor='black')
            
            ax.set_title(f'{metric} Comparison', fontsize=14, weight='bold', pad=15)
            ax.set_xlabel('Score', fontsize=12, weight='bold')
            ax.set_xlim(0, 1)
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f}', ha='left', va='center', fontsize=10, weight='bold')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: model_comparison.png")
        plt.close()
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (name, model_info) in enumerate(self.models.items()):
            cm = model_info['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'],
                       ax=axes[idx], cbar=True)
            
            axes[idx].set_title(f'{name}\nConfusion Matrix', fontsize=12, weight='bold')
            axes[idx].set_ylabel('Actual', fontsize=11, weight='bold')
            axes[idx].set_xlabel('Predicted', fontsize=11, weight='bold')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: confusion_matrices.png")
        plt.close()
    
    def generate_classification_reports(self):
        """Generate detailed classification reports"""
        print("\n" + "="*70)
        print("DETAILED CLASSIFICATION REPORTS")
        print("="*70)
        
        for name, model_info in self.models.items():
            print(f"\n{name}")
            print("-" * 70)
            report = classification_report(
                self.y_test, 
                model_info['y_pred'],
                target_names=['Negative', 'Positive'],
                digits=4
            )
            print(report)
    
    def save_best_model(self, best_model_name):
        """Save the best model and vectorizer"""
        print("\n" + "="*70)
        print("SAVING BEST MODEL")
        print("="*70)
        
        best_model = self.models[best_model_name]['model']
        
        # Save model
        with open('sentiment_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        print(f"✓ Saved model: sentiment_model.pkl")
        
        # Save vectorizer
        with open('tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"✓ Saved vectorizer: tfidf_vectorizer.pkl")
        
        # Save model metadata
        metadata = {
            'model_name': best_model_name,
            'train_accuracy': self.models[best_model_name]['train_accuracy'],
            'test_accuracy': self.models[best_model_name]['test_accuracy'],
            'precision': self.models[best_model_name]['precision'],
            'recall': self.models[best_model_name]['recall'],
            'f1_score': self.models[best_model_name]['f1_score'],
            'features': self.X_train_vec.shape[1],
            'training_samples': len(self.X_train),
            'test_samples': len(self.X_test)
        }
        
        with open('model_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✓ Saved metadata: model_metadata.pkl")
        
        return metadata
    
    def test_model_predictions(self, best_model_name):
        """Test model with sample predictions"""
        print("\n" + "="*70)
        print("SAMPLE PREDICTIONS TEST")
        print("="*70)
        
        model = self.models[best_model_name]['model']
        
        # Test samples
        test_samples = [
            "This product is absolutely amazing! Best purchase ever!",
            "Terrible quality, waste of money. Very disappointed.",
            "The phone camera is excellent and battery lasts all day",
            "Worst customer service. Product arrived damaged and broken.",
            "Love the design and performance. Highly recommended!",
            "Poor quality, overpriced, and bad delivery experience"
        ]
        
        print("\nTesting with sample reviews:")
        print("-" * 70)
        
        for i, sample in enumerate(test_samples, 1):
            # Preprocess (simplified for demo)
            sample_processed = sample.lower()
            
            # Vectorize
            sample_vec = self.vectorizer.transform([sample_processed])
            
            # Predict
            prediction = model.predict(sample_vec)[0]
            proba = model.predict_proba(sample_vec)[0]
            
            sentiment = "Positive" if prediction == 1 else "Negative"
            confidence = proba[prediction] * 100
            
            print(f"\n{i}. Review: {sample}")
            print(f"   Predicted: {sentiment} (Confidence: {confidence:.1f}%)")


def create_evaluation_summary(metadata, results_df):
    """Create evaluation summary report"""
    
    summary = []
    summary.append("="*70)
    summary.append("MODEL EVALUATION SUMMARY")
    summary.append("="*70)
    summary.append("")
    
    summary.append("BEST MODEL DETAILS")
    summary.append("-" * 70)
    summary.append(f"Model: {metadata['model_name']}")
    summary.append(f"Training Samples: {metadata['training_samples']:,}")
    summary.append(f"Test Samples: {metadata['test_samples']:,}")
    summary.append(f"Number of Features: {metadata['features']:,}")
    summary.append("")
    
    summary.append("PERFORMANCE METRICS")
    summary.append("-" * 70)
    summary.append(f"Training Accuracy: {metadata['train_accuracy']:.4f}")
    summary.append(f"Test Accuracy:     {metadata['test_accuracy']:.4f}")
    summary.append(f"Precision:         {metadata['precision']:.4f}")
    summary.append(f"Recall:            {metadata['recall']:.4f}")
    summary.append(f"F1 Score:          {metadata['f1_score']:.4f}")
    summary.append("")
    
    summary.append("ALL MODELS COMPARISON")
    summary.append("-" * 70)
    summary.append(results_df.to_string(index=False))
    summary.append("")
    
    summary.append("="*70)
    summary.append("END OF EVALUATION SUMMARY")
    summary.append("="*70)
    
    # Save summary
    with open('model_evaluation_summary.txt', 'w') as f:
        f.write('\n'.join(summary))
    
    print("\n✓ Saved: model_evaluation_summary.txt")
    
    # Print to console
    print("\n")
    print('\n'.join(summary))


def main():
    """Main execution function"""
    
    # Load processed data
    print("Loading processed data...")
    df = pd.read_csv('cleaned_data_processed.csv')
    print(f"Loaded {len(df)} reviews\n")
    
    # Initialize trainer
    trainer = SentimentModelTrainer(df)
    
    # Prepare data
    trainer.prepare_data(test_size=0.3, random_state=42)
    
    # Train models
    trainer.train_models()
    
    # Compare models
    best_model_name, results_df = trainer.compare_models()
    
    # Generate classification reports
    trainer.generate_classification_reports()
    
    # Create visualizations
    trainer.plot_model_comparison(results_df)
    trainer.plot_confusion_matrices()
    
    # Save best model
    metadata = trainer.save_best_model(best_model_name)
    
    # Test predictions
    trainer.test_model_predictions(best_model_name)
    
    # Create evaluation summary
    create_evaluation_summary(metadata, results_df)
    
    print("\n✓ Model training and evaluation complete!")


if __name__ == "__main__":
    main()

