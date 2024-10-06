from flask import Flask, render_template, request
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

app = Flask(__name__)

# Load dataset and preprocess (same as before)
df = pd.read_csv('Dataset-SA.csv')
df['Predicted_Sentiment'] = df['Review'].apply(lambda x: 'neutral' if pd.isna(x) else 'positive' if sia.polarity_scores(x)['compound'] > 0 else 'negative' if sia.polarity_scores(x)['compound'] < 0 else 'neutral')

# Inverted Index (same as before)
inverted_index = {}

def build_inverted_index():
    for index, row in df.iterrows():
        # Check if 'Review' is a valid string and not NaN
        if pd.notna(row['Review']):
            for word in row['Review'].split():
                word = word.lower()
                if word not in inverted_index:
                    inverted_index[word] = []
                inverted_index[word].append(index)

build_inverted_index()

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST' and 'keyword' in request.form:
        keyword = request.form['keyword'].lower()
        sentiment_filter = request.form['sentiment']
        
        indices = inverted_index.get(keyword, [])
        filtered_df = df.loc[indices]

        if sentiment_filter:
            filtered_df = filtered_df[filtered_df['Sentiment'] == sentiment_filter]

        results = filtered_df[['product_name', 'product_price', 'Rate', 'Review', 'Sentiment']]

    return render_template('index.html', tables=[results.to_html(classes='data')] if results is not None else None)

@app.route('/predict', methods=['GET', 'POST'])
def predict_sentiment():
    predicted_sentiment = None
    if request.method == 'POST' and 'user_input' in request.form:
        user_input = request.form['user_input']
        if user_input:
            score = sia.polarity_scores(user_input)['compound']
            if score > 0:
                predicted_sentiment = 'positive'
            elif score < 0:
                predicted_sentiment = 'negative'
            else:
                predicted_sentiment = 'neutral'

    return render_template('predict.html', sentiment=predicted_sentiment)

if __name__ == '__main__':
    app.run(debug=True)
