from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
import pandas as pd
from keras.preprocessing.text import tokenizer_from_json
import json

app=Flask(__name__)
model = load_model(r'C:\Users\super\OneDrive\桌面\adcademic\2000 ML\final project\Beauty_101_app\Beauty_101_app\model_filename.h5')

with open(r'C:\Users\super\OneDrive\桌面\adcademic\2000 ML\final project\Beauty_101_app\Beauty_101_app\tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

from keras.preprocessing.sequence import pad_sequences
#text = pd.read_csv(r'C:\Users\super\OneDrive\桌面\adcademic\2000 ML\final project\Beauty_101_app\Beauty_101_app\test.csv')
#text_collection = text['review_title_and_text']

def preprocess_text(text, tokenizer, max_seq_length):
    sequences = tokenizer.texts_to_sequences(text)
    padded_sequences = pad_sequences(sequences, maxlen=max_seq_length)
    return padded_sequences

def predict_sentiment(processed_text, model):
    prediction = model.predict(processed_text)
    return prediction


def softmax2label(prediction):
  prediction_label = "Positive" if prediction >= 0.5 else "Negative"
  return prediction_label

@app.route('/')
def inputs():
    return render_template('index.html')

@app.route('/sentiment',methods=['POST'])
def sentiment():
    if request.method == 'POST':
        review= request.form.get("user")
        processed_text = preprocess_text(review, tokenizer, 604)
        sentiment = predict_sentiment(processed_text, model)
        feelings = softmax2label(sentiment[0][0])
    return render_template('index.html', text=review, sentiment=feelings)

@app.route('/analysis',methods=['POST'])
def classify():
    file = request.files['file']
    if 'file' not in request.files:
        return 'No file part'

    if file.filename == '':
        return 'No selected file'
    if file:
        text_collection = pd.read_csv(file)
        text_collection['review_text'] = text_collection['review_text'].astype(str)
        pos = 0
        neg = 0
        for text in text_collection['review_text'][155:170]:
            processed_text = preprocess_text(text, tokenizer, 604)
            sentiment = predict_sentiment(processed_text, model)            
            x = softmax2label(sentiment[0][0])
            print(text)
            print(x)
            if x == "Positive":
                pos += 1
            else:
                neg += 1


        return render_template('index.html',positive=pos, negative=neg)

if __name__=='__main__':
    app.run()