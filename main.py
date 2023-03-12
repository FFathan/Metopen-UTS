import streamlit as st
import pandas as pd
import numpy as np

import pickle
import tensorflow as tf
import re
import contractions
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords.words('english')
nltk.download('wordnet')
stop_words = set(stopwords.words("english"))
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open("labelencoder.pickle", "rb") as f:
    le = pickle.load(f)
# le = LabelEncoder()
# le.fit(['pride', 'curiosity', 'sadness', 'disappointment', 'desire', 'surprise', 'confusion', 'remorse', 'excitement', 'grief', 'disapproval', 'love', 'neutral', 'annoyance', 'fear', 'approval', 'embarrassment', 'gratitude', 'nervousness', 'admiration', 'optimism', 'joy', 'relief', 'amusement', 'anger', 'disgust', 'realization', 'caring'])

imported_model_path = "lstm_True_False_50_32_0.001_5_256_pre_0.1_0.0.h5"
model = tf.keras.models.load_model(imported_model_path)
df_maxlen = 357
df_truncating = "pre"

def clean_text(text):
#   re_number = re.compile('[0-9]+')
  re_url = re.compile("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
#   re_tag = re.compile('\[[A-Z]+\]')
  re_char = re.compile('[^0-9a-zA-Z\s?!.,:\'\"//]+')
  re_char_clean = re.compile('[^0-9a-zA-Z\s?!.,\[\]]')
  re_punc = re.compile('[?!,.\'\"]')
  
  text = re.sub(re_char, "", text) # Remove unknown character 
  text = contractions.fix(text) # Expand contraction
  text = re.sub(re_url, ' [url] ', text) # Replace URL with number
  text = re.sub(re_char_clean, "", text) # Only alphanumeric and punctuations.
  text = re.sub(re_punc, "", text) # Remove punctuation.
  text = text.lower() # Lower text
  text = " ".join([w for w in text.split(' ') if w != " "]) # Remove whitespace

  return text

def lemmatization(text):
    lemmatizer= WordNetLemmatizer()

    text = text.split()
    text=[lemmatizer.lemmatize(y) for y in text]
    
    return " " .join(text)

def remove_stop_words(text):
    Text=[i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def Removing_numbers(text):
    text=''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):
    text = text.split()
    text=[y.lower() for y in text]
    
    return " " .join(text)

def Removing_punctuations(text):
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )
    
    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()

def Removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan
            
def normalize_text(df):
    df.Text=df.Text.apply(lambda text : lower_case(text))
    df.Text=df.Text.apply(lambda text : remove_stop_words(text))
    df.Text=df.Text.apply(lambda text : Removing_numbers(text))
    df.Text=df.Text.apply(lambda text : Removing_punctuations(text))
    df.Text=df.Text.apply(lambda text : Removing_urls(text))
    df.Text=df.Text.apply(lambda text : lemmatization(text))
    return df

def normalized_sentence(sentence):
    sentence= lower_case(sentence)
    sentence= remove_stop_words(sentence)
    sentence= Removing_numbers(sentence)
    sentence= Removing_punctuations(sentence)
    sentence= Removing_urls(sentence)
    sentence= lemmatization(sentence)
    return sentence

with st.form(key="form1", clear_on_submit=True):
    st.title('EMOTION SENTIMENT CLASSIFICATION ON GO EMOTIONS USING LSTM METHOD')
    st.text_input("Enter the text you want to classify", key="name")
    submitted = st.form_submit_button("Submit")
    if submitted:
        sentence = st.session_state.name
        st.write(sentence)
        sentence = clean_text(sentence)
        sentence = normalized_sentence(sentence)
        sentence = tokenizer.texts_to_sequences([sentence])
        sentence = pad_sequences(sentence, maxlen=df_maxlen, truncating=df_truncating)
        result = model.predict(sentence)[0]
        best_three_indices = np.argsort(result)[::-1][:3]
        label_best_three = le.inverse_transform(best_three_indices)
        result_best_three = result[best_three_indices]
        for i in range(3):
            st.write(f"{label_best_three[i]}: {'{:.4%}'.format(result_best_three[i])}")
