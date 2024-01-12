# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 22:45:33 2024

@author: sbkum
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def sentiment_prediction(sentence):
    df = pd.read_csv('data.csv')
    model = load_model('itStudio.h5')

    X_train, X_test, y_train, y_test = train_test_split(df.Sentence, df.Sentiment, test_size=0.2, random_state=42)
    token = Tokenizer(num_words=None)
    token.fit_on_texts(list(X_train) + list(X_test))
    word_index = token.word_index
    max_sequence_length = 300
    sequence = token.texts_to_sequences([sentence])
    sequence_padded = pad_sequences(sequence, maxlen=max_sequence_length, padding='pre', truncating='post')
    prediction = model.predict(sequence_padded)
    predicted_class_index = int(prediction.argmax(axis=-1))
    sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    predicted_sentiment = sentiment_labels[predicted_class_index]
    return predicted_sentiment

def main():
    # Title for the web app
    st.title("News Sentiment Analysis")
    
    #getting data
    statement = st.text_input("Enter the statement")
    
    #code for prediction 
    sentiment = 'Failed'
    
    #creating a button for prediction
    if st.button('Sentiment of Statement'):
        sentiment = sentiment_prediction(statement)
        
    st.success(sentiment)

if __name__ == '__main__':
    main()
 
