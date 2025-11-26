"""
Exploratory Data Analysis and Visualization Module
Generates business insights and creates visualization dashboard
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class SentimentAnalyzer:
    """Class for analyzing sentiment data and generating insights"""
    
    def __init__(self, df):
        self.df = df
        self.insights = []
        
    def generate_insights(self):
        """Generate business-relevant insights"""
        print("="*70)
        print("BUSINESS INSIGHTS GENERATION")
        print("="*70)
        
        # Insight 1: Overall sentiment distribution
        sentiment_dist = self.df['sentiment'].value_counts(normalize=True) * 100
        self.insights.append({
            'title': 'Overall Customer Satisfaction',
            'finding': f"Overall, {sentiment_dist['Positive']:.1f}% of reviews are positive, "
                      f"while {sentiment_dist['Negative']:.1f}% are negative. "
                      f"This indicates {'strong' if sentiment_dist['Positive'] > 70 else 'moderate'} customer satisfaction."
        })
        
        # Insight 2: Product comparison
        product_sentiment = pd.crosstab(self.df['product'], self.df['sentiment'], normalize='index') * 100
        best_product = product_sentiment['Positive'].idxmax()
        worst_product = product_sentiment['Positive'].idxmin()
        
        self.insights.append({
            'title': 'Product Performance Comparison',
            'finding': f"{best_product} has the highest positive sentiment at {product_sentiment.loc[best_product, 'Positive']:.1f}%, "
                      f"while {worst_product} has the lowest at {product_sentiment.loc[worst_product, 'Positive']:.1f}%. "
                      f"Difference: {product_sentiment.loc[best_product, 'Positive'] - product_sentiment.loc[worst_product, 'Positive']:.1f} percentage points."
        })
        
        # Insight 3: Common complaints identification
        negative_reviews = self.df[self.df['sentiment'] == 'Negative']['review_text'].str.lower()
        
        # Define complaint keywords
        complaint_keywords = {
            'battery': ['battery', 'charge', 'charging'],
            'delivery': ['delivery', 'shipping', 'delivered', 'package'],
            'quality': ['quality', 'defect', 'broken', 'poor'],
            'price': ['price', 'expensive', 'cost', 'money'],
            'camera': ['camera', 'photo', 'picture'],
            'heating': ['heat', 'heating', 'hot', 'warm'],
            'customer_service': ['service', 'support', 'customer care', 'refund']
        }
        
        complaint_counts = {}
        total_negative = len(negative_reviews)
        
        for category, keywords in complaint_keywords.items():
            count = sum(negative_reviews.str.contains('|'.join(keywords), na=False))
            complaint_counts[category] = (count / total_negative * 100) if total_negative > 0 else 0
        
        # Sort by frequency
        top_complaints = sorted(complaint_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        complaints_text = ", ".join([f"{cat.replace('_', ' ').title()} ({pct:.1f}%)" 
                                     for cat, pct in top_complaints])
        
        self.insights.append({
            'title': 'Most Common Customer Complaints',
            'finding': f"The top 3 complaints mentioned in negative reviews are: {complaints_text}. "
                      f"Addressing these issues could significantly improve customer satisfaction."
        })
        
        # Insight 4: Review length and sentiment correlation
        avg_word_positive = self.df[self.df['sentiment'] == 'Positive']['word_count'].mean()
        avg_word_negative = self.df[self.df['sentiment'] == 'Negative']['word_count'].mean()
        
        longer_reviews = 'Negative' if avg_word_negative > avg_word_positive else 'Positive'
        diff_pct = abs(avg_word_negative - avg_word_positive) / min(avg_word_negative, avg_word_positive) * 100
        
        self.insights.append({
            'title': 'Review Length and Sentiment Relationship',
            'finding': f"{longer_reviews} reviews are on average {diff_pct:.1f}% longer "
                      f"(Positive: {avg_word_positive:.0f} words, Negative: {avg_word_negative:.0f} words). "
                      f"Customers with {'issues' if longer_reviews == 'Negative' else 'positive experiences'} "
                      f"tend to write more detailed reviews."
        })
        
        # Insight 5: Temporal trend (if date available)
        if 'year_month' in self.df.columns and self.df['year_month'].notna().any():
            monthly_sentiment = self.df.groupby(['year_month', 'sentiment']).size().unstack(fill_value=0)
            if 'Positive' in monthly_sentiment.columns and 'Negative' in monthly_sentiment.columns:
                monthly_sentiment['pos_ratio'] = (monthly_sentiment['Positive'] / 
                                                  (monthly_sentiment['Positive'] + monthly_sentiment['Negative']) * 100)
                
                trend = monthly_sentiment['pos_ratio'].diff().mean()
                trend_direction = 'improving' if trend > 0 else 'declining'
                
                self.insights.append({
                    'title': 'Sentiment Trend Over Time',
                    'finding': f"Customer sentiment is {trend_direction} over time "
                              f"({'increasing' if trend > 0 else 'decreasing'} by {abs(trend):.2f}% per month on average). "
                              f"This suggests product/service quality is {'getting better' if trend > 0 else 'deteriorating'}."
                })
        
        # Insight 6: Rating distribution
        rating_dist = self.df['star_rating'].value_counts().sort_index()
        mode_rating = self.df['star_rating'].mode()[0]
        median_rating = self.df['star_rating'].median()
        
        self.insights.append({
            'title': 'Rating Distribution Insights',
            'finding': f"Most common rating is {mode_rating} stars, with median rating of {median_rating}. "
                      f"{(self.df['star_rating'] == 5).sum()} reviews ({(self.df['star_rating'] == 5).sum() / len(self.df) * 100:.1f}%) "
                      f"gave perfect 5-star ratings, indicating {'strong' if (self.df['star_rating'] == 5).sum() / len(self.df) > 0.4 else 'moderate'} brand loyalty."
        })
        
        # Print insights
        print("\n")
        for i, insight in enumerate(self.insights, 1):
            print(f"💡 INSIGHT {i}: {insight['title']}")
            print(f"   {insight['finding']}")
            print()
        
        return self.insights
    
    def identify_pain_points(self):
        """Identify specific pain points from negative reviews"""
        print("="*70)
        print("PAIN POINTS ANALYSIS")
        print("="*70)
        
        negative_text = ' '.join(self.df[self.df['sentiment'] == 'Negative']['processed_text'].values)
        
        # Extract most common words
        words = negative_text.split()
        word_freq = Counter(words)
        
        # Remove very common generic words
        generic_words = {'phone', 'product', 'shoes', 'iphone', 'nike', 'flipkart', 'order', 'bought'}
        filtered_freq = {word: count for word, count in word_freq.items() 
                         if word not in generic_words and len(word) > 3}
        
        # Get top pain points
        top_pain_points = Counter(filtered_freq).most_common(15)
        
        print("\nTop 15 Pain Points (from negative reviews):")
        for i, (word, count) in enumerate(top_pain_points, 1):
            print(f"{i:2d}. {word:20s} - mentioned {count:3d} times")
        
        return top_pain_points
    
    def identify_praise_points(self):
        """Identify what customers praise from positive reviews"""
        print("\n" + "="*70)
        print("PRAISE POINTS ANALYSIS")
        print("="*70)
        
        positive_text = ' '.join(self.df[self.df['sentiment'] == 'Positive']['processed_text'].values)
        
        # Extract most common words
        words = positive_text.split()
        word_freq = Counter(words)
        
        # Remove very common generic words
        generic_words = {'phone', 'product', 'shoes', 'iphone', 'nike', 'flipkart', 'order', 'bought', 'got'}
        filtered_freq = {word: count for word, count in word_freq.items() 
                         if word not in generic_words and len(word) > 3}
        
        # Get top praise points
        top_praise_points = Counter(filtered_freq).most_common(15)
        
        print("\nTop 15 Praise Points (from positive reviews):")
        for i, (word, count) in enumerate(top_praise_points, 1):
            print(f"{i:2d}. {word:20s} - mentioned {count:3d} times")
        
        return top_praise_points


class VisualizationDashboard:
    """Class for creating visualization dashboard"""
    
    def __init__(self, df):
        self.df = df
        
    def create_dashboard(self):
        """Create complete visualization dashboard"""
        print("\n" + "="*70)
        print("CREATING VISUALIZATION DASHBOARD")
        print("="*70)
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Sentiment Distribution (Pie Chart)
        ax1 = plt.subplot(2, 3, 1)
        self.plot_sentiment_distribution(ax1)
        
        # 2. Sentiment by Product (Bar Chart)
        ax2 = plt.subplot(2, 3, 2)
        self.plot_sentiment_by_product(ax2)
        
        # 3. Rating Distribution (Histogram)
        ax3 = plt.subplot(2, 3, 3)
        self.plot_rating_distribution(ax3)
        
        # 4. Sentiment Trend Over Time (Line Plot)
        ax4 = plt.subplot(2, 3, 4)
        self.plot_sentiment_trend(ax4)
        
        # 5. Review Length Distribution
        ax5 = plt.subplot(2, 3, 5)
        self.plot_review_length_distribution(ax5)
        
        # 6. Top Pain Points (Bar Chart)
        ax6 = plt.subplot(2, 3, 6)
        self.plot_pain_points(ax6)
        
        plt.tight_layout()
        plt.savefig('sentiment_dashboard.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: sentiment_dashboard.png")
        plt.close()
        
        # Create word clouds separately (larger images)
        self.create_word_clouds()
        
        print("\n✓ All visualizations created successfully!")
        
    def plot_sentiment_distribution(self, ax):
        """Plot sentiment distribution pie chart"""
        sentiment_counts = self.df['sentiment'].value_counts()
        colors = ['#2ecc71', '#e74c3c']  # Green for Positive, Red for Negative
        
        wedges, texts, autotexts = ax.pie(sentiment_counts.values, 
                                           labels=sentiment_counts.index,
                                           autopct='%1.1f%%',
                                           colors=colors,
                                           startangle=90,
                                           textprops={'fontsize': 12, 'weight': 'bold'})
        
        ax.set_title('Overall Sentiment Distribution', fontsize=14, weight='bold', pad=20)
        
    def plot_sentiment_by_product(self, ax):
        """Plot sentiment by product"""
        product_sentiment = pd.crosstab(self.df['product'], self.df['sentiment'])
        product_sentiment.plot(kind='bar', ax=ax, color=['#e74c3c', '#2ecc71'], width=0.7)
        
        ax.set_title('Sentiment Distribution by Product', fontsize=14, weight='bold', pad=20)
        ax.set_xlabel('Product', fontsize=12, weight='bold')
        ax.set_ylabel('Number of Reviews', fontsize=12, weight='bold')
        ax.legend(title='Sentiment', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        
    def plot_rating_distribution(self, ax):
        """Plot rating distribution histogram"""
        rating_counts = self.df['star_rating'].value_counts().sort_index()
        
        colors_map = {1: '#e74c3c', 2: '#e67e22', 3: '#f39c12', 4: '#3498db', 5: '#2ecc71'}
        colors = [colors_map[rating] for rating in rating_counts.index]
        
        ax.bar(rating_counts.index, rating_counts.values, color=colors, width=0.6, edgecolor='black')
        ax.set_title('Rating Distribution', fontsize=14, weight='bold', pad=20)
        ax.set_xlabel('Star Rating', fontsize=12, weight='bold')
        ax.set_ylabel('Number of Reviews', fontsize=12, weight='bold')
        ax.set_xticks(rating_counts.index)
        ax.grid(axis='y', alpha=0.3)
        
    def plot_sentiment_trend(self, ax):
        """Plot sentiment trend over time"""
        if 'year_month' in self.df.columns and self.df['year_month'].notna().any():
            # Filter out NaT values
            df_with_dates = self.df[self.df['year_month'].notna()].copy()
            
            if len(df_with_dates) > 0:
                monthly_sentiment = df_with_dates.groupby(['year_month', 'sentiment']).size().unstack(fill_value=0)
                
                if 'Positive' in monthly_sentiment.columns and 'Negative' in monthly_sentiment.columns:
                    monthly_sentiment['total'] = monthly_sentiment.sum(axis=1)
                    monthly_sentiment['positive_pct'] = (monthly_sentiment['Positive'] / monthly_sentiment['total'] * 100)
                    monthly_sentiment['negative_pct'] = (monthly_sentiment['Negative'] / monthly_sentiment['total'] * 100)
                    
                    # Convert period to string for plotting
                    x = range(len(monthly_sentiment))
                    labels = [str(period) for period in monthly_sentiment.index]
                    
                    ax.plot(x, monthly_sentiment['positive_pct'], marker='o', color='#2ecc71', 
                           linewidth=2, label='Positive', markersize=6)
                    ax.plot(x, monthly_sentiment['negative_pct'], marker='o', color='#e74c3c', 
                           linewidth=2, label='Negative', markersize=6)
                    
                    ax.set_title('Sentiment Trend Over Time', fontsize=14, weight='bold', pad=20)
                    ax.set_xlabel('Month', fontsize=12, weight='bold')
                    ax.set_ylabel('Percentage (%)', fontsize=12, weight='bold')
                    ax.legend(fontsize=10)
                    ax.grid(True, alpha=0.3)
                    
                    # Set x-axis labels
                    step = max(1, len(labels) // 10)
                    ax.set_xticks(x[::step])
                    ax.set_xticklabels(labels[::step], rotation=45, ha='right')
                else:
                    ax.text(0.5, 0.5, 'Insufficient data for trend analysis', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
            else:
                ax.text(0.5, 0.5, 'No date information available', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
        else:
            ax.text(0.5, 0.5, 'No date information available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
    def plot_review_length_distribution(self, ax):
        """Plot review length distribution by sentiment"""
        positive_lengths = self.df[self.df['sentiment'] == 'Positive']['word_count']
        negative_lengths = self.df[self.df['sentiment'] == 'Negative']['word_count']
        
        ax.hist([positive_lengths, negative_lengths], bins=30, 
               label=['Positive', 'Negative'], color=['#2ecc71', '#e74c3c'], 
               alpha=0.7, edgecolor='black')
        
        ax.set_title('Review Length Distribution by Sentiment', fontsize=14, weight='bold', pad=20)
        ax.set_xlabel('Word Count', fontsize=12, weight='bold')
        ax.set_ylabel('Frequency', fontsize=12, weight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
    def plot_pain_points(self, ax):
        """Plot top pain points"""
        negative_text = ' '.join(self.df[self.df['sentiment'] == 'Negative']['processed_text'].values)
        words = negative_text.split()
        word_freq = Counter(words)
        
        # Remove generic words
        generic_words = {'phone', 'product', 'shoes', 'iphone', 'nike', 'flipkart', 'order', 'bought', 'got'}
        filtered_freq = {word: count for word, count in word_freq.items() 
                         if word not in generic_words and len(word) > 3}
        
        # Get top 10
        top_10 = dict(Counter(filtered_freq).most_common(10))
        
        words = list(top_10.keys())
        counts = list(top_10.values())
        
        bars = ax.barh(words, counts, color='#e74c3c', edgecolor='black')
        ax.set_title('Top 10 Pain Points (Negative Reviews)', fontsize=14, weight='bold', pad=20)
        ax.set_xlabel('Frequency', fontsize=12, weight='bold')
        ax.set_ylabel('Keywords', fontsize=12, weight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{int(width)}', ha='left', va='center', fontsize=9, weight='bold')
        
    def create_word_clouds(self):
        """Create word clouds for positive and negative reviews"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Positive word cloud
        positive_text = ' '.join(self.df[self.df['sentiment'] == 'Positive']['processed_text'].values)
        
        wordcloud_pos = WordCloud(width=800, height=400, 
                                  background_color='white',
                                  colormap='Greens',
                                  max_words=100,
                                  relative_scaling=0.5,
                                  min_font_size=10).generate(positive_text)
        
        ax1.imshow(wordcloud_pos, interpolation='bilinear')
        ax1.axis('off')
        ax1.set_title('Positive Reviews - Word Cloud', fontsize=16, weight='bold', pad=20)
        
        # Negative word cloud
        negative_text = ' '.join(self.df[self.df['sentiment'] == 'Negative']['processed_text'].values)
        
        wordcloud_neg = WordCloud(width=800, height=400,
                                  background_color='white',
                                  colormap='Reds',
                                  max_words=100,
                                  relative_scaling=0.5,
                                  min_font_size=10).generate(negative_text)
        
        ax2.imshow(wordcloud_neg, interpolation='bilinear')
        ax2.axis('off')
        ax2.set_title('Negative Reviews - Word Cloud', fontsize=16, weight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig('wordclouds.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: wordclouds.png")
        plt.close()


def generate_insights_report(df, insights, pain_points, praise_points):
    """Generate a text report of insights"""
    
    report = []
    report.append("="*70)
    report.append("SENTIMENT ANALYSIS INSIGHTS REPORT")
    report.append("="*70)
    report.append("")
    
    # Dataset Overview
    report.append("DATASET OVERVIEW")
    report.append("-" * 70)
    report.append(f"Total Reviews Analyzed: {len(df):,}")
    report.append(f"Products: {', '.join(df['product'].unique())}")
    try:
        if 'date' in df.columns and df['date'].notna().any():
            report.append(f"Date Range: {df['date'].min()} to {df['date'].max()}")
        else:
            report.append("Date Range: Not available")
    except:
        report.append("Date Range: Not available")
    report.append("")
    
    # Sentiment Summary
    report.append("SENTIMENT SUMMARY")
    report.append("-" * 70)
    sentiment_pct = df['sentiment'].value_counts(normalize=True) * 100
    for sentiment, pct in sentiment_pct.items():
        report.append(f"{sentiment}: {pct:.1f}% ({df['sentiment'].value_counts()[sentiment]:,} reviews)")
    report.append("")
    
    # Product-wise Performance
    report.append("PRODUCT-WISE PERFORMANCE")
    report.append("-" * 70)
    for product in df['product'].unique():
        product_df = df[df['product'] == product]
        pos_pct = (product_df['sentiment'] == 'Positive').sum() / len(product_df) * 100
        avg_rating = product_df['star_rating'].mean()
        report.append(f"{product}:")
        report.append(f"  - Positive Sentiment: {pos_pct:.1f}%")
        report.append(f"  - Average Rating: {avg_rating:.2f}/5.0")
        report.append(f"  - Total Reviews: {len(product_df):,}")
        report.append("")
    
    # Business Insights
    report.append("KEY BUSINESS INSIGHTS")
    report.append("-" * 70)
    for i, insight in enumerate(insights, 1):
        report.append(f"{i}. {insight['title']}")
        report.append(f"   {insight['finding']}")
        report.append("")
    
    # Pain Points
    report.append("TOP CUSTOMER PAIN POINTS (from negative reviews)")
    report.append("-" * 70)
    for i, (word, count) in enumerate(pain_points[:10], 1):
        report.append(f"{i:2d}. {word:20s} - mentioned {count:3d} times")
    report.append("")
    
    # Praise Points
    report.append("TOP CUSTOMER PRAISE POINTS (from positive reviews)")
    report.append("-" * 70)
    for i, (word, count) in enumerate(praise_points[:10], 1):
        report.append(f"{i:2d}. {word:20s} - mentioned {count:3d} times")
    report.append("")
    
    # Recommendations
    report.append("RECOMMENDATIONS")
    report.append("-" * 70)
    report.append("1. Focus on addressing the top pain points identified in negative reviews")
    report.append("2. Leverage positive aspects mentioned in praise points for marketing")
    report.append("3. Monitor sentiment trends over time to catch issues early")
    report.append("4. Product-specific improvements based on sentiment analysis")
    report.append("5. Implement feedback loops to continuously improve customer experience")
    report.append("")
    
    report.append("="*70)
    report.append("END OF REPORT")
    report.append("="*70)
    
    # Save report
    with open('insights_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print("\n✓ Saved: insights_report.txt")
    
    # Also print to console
    print("\n")
    print('\n'.join(report))


def main():
    """Main execution function"""
    
    # Load processed data
    print("Loading processed data...")
    df = pd.read_csv('cleaned_data_processed.csv')
    
    # Convert year_month back to period if it exists
    if 'year_month' in df.columns:
        try:
            df['year_month'] = pd.to_datetime(df['year_month']).dt.to_period('M')
        except:
            pass
    
    print(f"Loaded {len(df)} reviews\n")
    
    # Generate insights
    analyzer = SentimentAnalyzer(df)
    insights = analyzer.generate_insights()
    pain_points = analyzer.identify_pain_points()
    praise_points = analyzer.identify_praise_points()
    
    # Create visualizations
    dashboard = VisualizationDashboard(df)
    dashboard.create_dashboard()
    
    # Generate report
    generate_insights_report(df, insights, pain_points, praise_points)
    
    print("\n✓ EDA and visualization complete!")


if __name__ == "__main__":
    main()

