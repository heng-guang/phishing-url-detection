from flask import Flask, request, url_for, redirect, render_template, jsonify
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import pickle
import numpy as np
import regex as re
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split, GridSearchCV

# Initalise the Flask app
app = Flask(__name__)

# Loads pre-trained model
model = pickle.load(open('g1-phishing-link-detection.sav', 'rb'))

df = pd.read_csv('phishing_site_urls.csv')
df.columns = [x.lower() for x in df.columns]
df['Y'] = df['label'].apply(lambda x: 0 if x == 'good' else 1)

def url_process(url):
    url = re.sub('[^a-zA-Z\ \n]', '.', url.lower())
    url =  re.sub('\.{1,}', ' ', url)
    url = url.split(' ')
    
    stemmer = SnowballStemmer("english")
    url = [stemmer.stem(word) for word in url]
    url = ' '.join(url)
    return url
    
df['url_clean'] = df['url'].apply(url_process)
cv=CountVectorizer(ngram_range=(1, 2))
X_train, X_test, y_train, y_test = train_test_split(df['url_clean'], df['Y'], test_size=0.2, random_state=5555)
cv_fit = cv.fit(X_train)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    url = request.form.get('url')
    a = url_process(url)
    a_trans = cv_fit.transform(pd.Series(a))
    model_pred = model.predict(a_trans)
    
    if model_pred == 1:
        return render_template('home.html',pred='Warning! Site highly likely to be a phishing link! Proceed with caution!')
    else:
        return render_template('home.html',pred='Great! Site is likely to be safe!')

if __name__ == '__main__':
    app.run(debug=True)