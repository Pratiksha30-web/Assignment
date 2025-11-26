"""
Flask API for Sentiment Prediction
Accepts review text and returns sentiment prediction
"""

from flask import Flask, request, jsonify, render_template_string
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Initialize NLTK components
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

app = Flask(__name__)

# Load model and vectorizer
print("Loading model and vectorizer...")
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

print(f"✓ Loaded model: {metadata['model_name']}")
print(f"✓ Model accuracy: {metadata['test_accuracy']:.4f}")
print(f"✓ Model F1 Score: {metadata['f1_score']:.4f}")

# Initialize text processor
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words -= {'not', 'no', 'never', 'neither', 'nor', 'very', 'too', 'good', 'bad'}


def preprocess_text(text):
    """Preprocess review text"""
    if not text or text.strip() == '':
        return ''
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    processed_tokens = []
    for token in tokens:
        if token not in stop_words and len(token) > 2:
            lemmatized = lemmatizer.lemmatize(token)
            processed_tokens.append(lemmatized)
    
    # Join tokens back
    processed_text = ' '.join(processed_tokens)
    
    return processed_text


# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Sentiment Analysis API</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .model-info {
            background: #f8f9fa;
            padding: 20px;
            border-bottom: 2px solid #e9ecef;
        }
        
        .model-info h3 {
            color: #495057;
            margin-bottom: 15px;
        }
        
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }
        
        .metric {
            background: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .metric-label {
            font-size: 0.9em;
            color: #6c757d;
            margin-bottom: 5px;
        }
        
        .metric-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }
        
        .content {
            padding: 40px;
        }
        
        .form-group {
            margin-bottom: 25px;
        }
        
        label {
            display: block;
            font-weight: 600;
            color: #495057;
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            font-size: 1em;
            font-family: inherit;
            resize: vertical;
            transition: border-color 0.3s;
        }
        
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 1.1em;
            font-weight: 600;
            border-radius: 10px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            width: 100%;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        .result {
            margin-top: 30px;
            padding: 25px;
            border-radius: 10px;
            display: none;
            animation: slideIn 0.5s ease;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .result.positive {
            background: #d4edda;
            border: 2px solid #28a745;
        }
        
        .result.negative {
            background: #f8d7da;
            border: 2px solid #dc3545;
        }
        
        .result-header {
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 15px;
        }
        
        .result.positive .result-header {
            color: #155724;
        }
        
        .result.negative .result-header {
            color: #721c24;
        }
        
        .confidence-bar {
            background: #e9ecef;
            height: 30px;
            border-radius: 15px;
            overflow: hidden;
            margin: 15px 0;
        }
        
        .confidence-fill {
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            transition: width 0.5s ease;
        }
        
        .confidence-fill.positive {
            background: #28a745;
        }
        
        .confidence-fill.negative {
            background: #dc3545;
        }
        
        .examples {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        
        .examples h3 {
            color: #495057;
            margin-bottom: 15px;
        }
        
        .example {
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .example:hover {
            transform: translateX(5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .example-label {
            font-weight: 600;
            color: #667eea;
            margin-bottom: 5px;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎭 Sentiment Analysis API</h1>
            <p>Analyze customer review sentiment in real-time</p>
        </div>
        
        <div class="model-info">
            <h3>📊 Model Performance</h3>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-label">Model</div>
                    <div class="metric-value">{{ model_name }}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Accuracy</div>
                    <div class="metric-value">{{ accuracy }}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">F1 Score</div>
                    <div class="metric-value">{{ f1_score }}</div>
                </div>
            </div>
        </div>
        
        <div class="content">
            <form id="sentimentForm">
                <div class="form-group">
                    <label for="review">Enter Review Text:</label>
                    <textarea 
                        id="review" 
                        name="review" 
                        rows="6" 
                        placeholder="Type or paste a customer review here..."
                        required
                    ></textarea>
                </div>
                
                <button type="submit">🔍 Analyze Sentiment</button>
            </form>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing...</p>
            </div>
            
            <div class="result" id="result">
                <div class="result-header" id="resultHeader"></div>
                <div class="confidence-bar">
                    <div class="confidence-fill" id="confidenceFill"></div>
                </div>
                <p id="resultText"></p>
            </div>
            
            <div class="examples">
                <h3>💡 Try these examples:</h3>
                <div class="example" onclick="fillExample('Amazing product! Best purchase ever. Highly recommended!')">
                    <div class="example-label">Positive Example</div>
                    <div>Amazing product! Best purchase ever. Highly recommended!</div>
                </div>
                <div class="example" onclick="fillExample('Terrible quality. Waste of money. Very disappointed with this purchase.')">
                    <div class="example-label">Negative Example</div>
                    <div>Terrible quality. Waste of money. Very disappointed with this purchase.</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        function fillExample(text) {
            document.getElementById('review').value = text;
        }
        
        document.getElementById('sentimentForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const reviewText = document.getElementById('review').value;
            const resultDiv = document.getElementById('result');
            const loadingDiv = document.getElementById('loading');
            
            // Show loading
            resultDiv.style.display = 'none';
            loadingDiv.style.display = 'block';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ review_text: reviewText })
                });
                
                const data = await response.json();
                
                // Hide loading
                loadingDiv.style.display = 'none';
                
                if (data.status === 'success') {
                    // Show result
                    const sentiment = data.sentiment;
                    const confidence = data.confidence;
                    
                    resultDiv.className = 'result ' + sentiment.toLowerCase();
                    resultDiv.style.display = 'block';
                    
                    document.getElementById('resultHeader').textContent = 
                        sentiment === 'Positive' ? '😊 Positive Sentiment' : '😞 Negative Sentiment';
                    
                    const confidenceFill = document.getElementById('confidenceFill');
                    confidenceFill.className = 'confidence-fill ' + sentiment.toLowerCase();
                    confidenceFill.style.width = confidence + '%';
                    confidenceFill.textContent = confidence + '% confident';
                    
                    document.getElementById('resultText').textContent = 
                        `The model predicts this review is ${sentiment.toLowerCase()} with ${confidence}% confidence.`;
                } else {
                    alert('Error: ' + data.message);
                }
            } catch (error) {
                loadingDiv.style.display = 'none';
                alert('Error: ' + error.message);
            }
        });
    </script>
</body>
</html>
"""


@app.route('/')
def home():
    """Render home page"""
    return render_template_string(
        HTML_TEMPLATE,
        model_name=metadata['model_name'].split()[0],
        accuracy=f"{metadata['test_accuracy']:.2%}",
        f1_score=f"{metadata['f1_score']:.3f}"
    )


@app.route('/predict', methods=['POST'])
def predict():
    """Predict sentiment for a review"""
    try:
        # Get review text from request
        data = request.get_json()
        review_text = data.get('review_text', '')
        
        if not review_text or review_text.strip() == '':
            return jsonify({
                'status': 'error',
                'message': 'Review text is required'
            }), 400
        
        # Preprocess text
        processed_text = preprocess_text(review_text)
        
        if not processed_text or processed_text.strip() == '':
            return jsonify({
                'status': 'error',
                'message': 'Review text could not be processed'
            }), 400
        
        # Vectorize
        text_vectorized = vectorizer.transform([processed_text])
        
        # Predict
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]
        
        # Get sentiment and confidence
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = round(probabilities[prediction] * 100, 2)
        
        return jsonify({
            'status': 'success',
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': {
                'negative': round(probabilities[0] * 100, 2),
                'positive': round(probabilities[1] * 100, 2)
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': metadata['model_name'],
        'accuracy': metadata['test_accuracy'],
        'f1_score': metadata['f1_score']
    })


if __name__ == '__main__':
    print("\n" + "="*70)
    print("Starting Sentiment Analysis API")
    print("="*70)
    print(f"Model: {metadata['model_name']}")
    print(f"Accuracy: {metadata['test_accuracy']:.4f}")
    print(f"F1 Score: {metadata['f1_score']:.4f}")
    print("="*70)
    print("\nAPI is running on http://localhost:5000")
    print("Open your browser and go to http://localhost:5000\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)

