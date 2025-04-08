from flask import Flask, request, jsonify, render_template
import joblib
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files
import os

# Download dataset (you can replace this with your own dataset)
nltk.download('movie_reviews')
from nltk.corpus import movie_reviews

def load_data():
    docs, labels = [], []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            docs.append(movie_reviews.raw(fileid))
            labels.append(1 if category == 'pos' else 0)  # Positive = 1, Negative = 0
    return docs, labels

# Load and split dataset
docs, labels = load_data()
X_train, X_test, y_train, y_test = train_test_split(docs, labels, test_size=0.2, random_state=42)

# Create and train model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'sentiment_model.pkl')

# Load trained model
model = joblib.load('sentiment_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    sentiment_score = model.predict([text])[0]
    sentiment = "Positive" if sentiment_score == 1 else "Negative"

    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)