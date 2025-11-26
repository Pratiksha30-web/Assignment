"""
Master Script to Run Complete Sentiment Analysis Pipeline
Executes all steps in sequence: Data Cleaning -> EDA -> Model Training
"""

import subprocess
import sys
import time


def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")


def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print_header(f"STEP: {description}")
    print(f"Running: {script_name}")
    print("-" * 80)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed_time = time.time() - start_time
        print("-" * 80)
        print(f"✓ {description} completed successfully!")
        print(f"⏱  Time taken: {elapsed_time:.2f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print("-" * 80)
        print(f"✗ Error in {description}")
        print(f"⏱  Time taken: {elapsed_time:.2f} seconds")
        print(f"Error details: {e}")
        return False


def main():
    """Main execution function"""
    
    print_header("🚀 SENTIMENT ANALYSIS PIPELINE")
    print("This script will execute the complete sentiment analysis pipeline:")
    print("1. Data Cleaning and Preprocessing")
    print("2. Exploratory Data Analysis and Visualization")
    print("3. Model Training and Evaluation")
    print("\nPlease ensure you have:")
    print("  - iphone14_customer_review.csv")
    print("  - web_scraped.csv")
    print("  - All required packages installed (run: pip install -r requirements.txt)")
    
    input("\nPress Enter to continue...")
    
    pipeline_start = time.time()
    
    # Step 1: Data Cleaning
    success = run_script('data_cleaning.py', 'Data Cleaning and Preprocessing')
    if not success:
        print("\n❌ Pipeline stopped due to error in data cleaning")
        return
    
    # Step 2: EDA and Visualization
    success = run_script('eda_and_visualization.py', 'Exploratory Data Analysis and Visualization')
    if not success:
        print("\n❌ Pipeline stopped due to error in EDA")
        return
    
    # Step 3: Model Training
    success = run_script('model_training.py', 'Model Training and Evaluation')
    if not success:
        print("\n❌ Pipeline stopped due to error in model training")
        return
    
    # Pipeline Complete
    pipeline_time = time.time() - pipeline_start
    
    print_header("✅ PIPELINE EXECUTION COMPLETE!")
    
    print("📁 Generated Files:")
    print("   Data Files:")
    print("     - cleaned_data_raw.csv")
    print("     - cleaned_data_processed.csv")
    print("     - cleaned_data_all_sentiments.csv")
    print("\n   Model Files:")
    print("     - sentiment_model.pkl")
    print("     - tfidf_vectorizer.pkl")
    print("     - model_metadata.pkl")
    print("\n   Visualization Files:")
    print("     - sentiment_dashboard.png")
    print("     - wordclouds.png")
    print("     - model_comparison.png")
    print("     - confusion_matrices.png")
    print("\n   Report Files:")
    print("     - insights_report.txt")
    print("     - model_evaluation_summary.txt")
    
    print(f"\n⏱  Total pipeline execution time: {pipeline_time/60:.2f} minutes")
    
    print("\n" + "="*80)
    print("🎯 NEXT STEPS")
    print("="*80)
    print("\n1. Review the generated insights and visualizations")
    print("2. Check model performance metrics")
    print("3. Test the API locally:")
    print("   python app.py")
    print("   Then open: http://localhost:5000")
    print("\n4. Deploy using Docker:")
    print("   docker build -t sentiment-api .")
    print("   docker run -d -p 5000:5000 sentiment-api")
    print("   Then open: http://localhost:5000")
    
    print("\n✨ Thank you for using the Sentiment Analysis Pipeline!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

