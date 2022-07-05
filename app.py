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
cv = pickle.load(open("vector.pickel", "rb"))

def url_process(url):
    url = re.sub('[^a-zA-Z\ \n]', '.', url.lower())
    url =  re.sub('\.{1,}', ' ', url)
    url = url.split(' ')
    
    stemmer = SnowballStemmer("english")
    url = [stemmer.stem(word) for word in url]
    url = ' '.join(url)
    return url

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    url = request.form.get('url')
    a = url_process(url)
    a_trans = cv.transform(pd.Series(a))
    model_pred = model.predict(a_trans)
    
    if model_pred == 1:
        return render_template('home.html',pred='Warning! Site highly likely to be a phishing link! Proceed with caution!')
    else:
        return render_template('home.html',pred='Great! Site is likely to be safe!')

if __name__ == '__main__':
    app.run(debug=True)