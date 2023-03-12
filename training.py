
dataset_path = 'go_emotions_dataset.csv'

use_nltk_preprocessing = True

use_glove = False
path_to_glove_file = 'glove.6B.50d.txt'

importing_model = False
# imported_model_path = "Model/lstm_True_False_50_32_0.001_5_256_pre_0.1_0.0"
imported_model_path = "Model/lstm_True_False_50_32_0.001_5_256_pre_0.1_0.0.h5"


embedding_dim = 50
lstm_dim = 32

learning_rate = 0.001
training_epoch = 5
training_batch_size = 256

df_truncating = 'pre'
lstm_dropout = 0.1
lstm_recurrent_dropout = 0.0

exported_model_path = 'lstm_' + str(use_nltk_preprocessing) + "_" \
    + str(use_glove) + "_" + str(embedding_dim) + "_" \
    + str(lstm_dim) + "_" + str(learning_rate) + "_" \
    + str(training_epoch) + "_" + str(training_batch_size) + "_" \
    + str(df_truncating) + "_" + str(lstm_dropout) + "_" \
    + str(lstm_recurrent_dropout)
exported_model_path_h5= exported_model_path + ".h5"

train_dataset_path = 'train_dataset.csv'
validation_dataset_path = 'validation_dataset.csv'
test_dataset_path = 'test_dataset.csv'

sentences = [
            "He's over the moon about being accepted to the university",
            "Your point on this certain matter made me outrageous, how can you say so? This is insane.",
            "I can't do it, I'm not ready to lose anything, just leave me alone",
            "Merlin's beard harry, you can cast the Patronus charm! I'm amazed!"
            ]
#################################################################

import re
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional
from tqdm import tqdm

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords.words('english')

sns.set(font_scale=1.3)
nltk.download('omw-1.4')

tqdm.pandas()

data = pd.read_csv(dataset_path)
emotions = set(data.columns[3:])

# positive = {'admiration','amusement','approval','caring','desire','excitement','gratitude','joy','love','optimism','pride','relief'}
# negative = {'sadness','fear','embarrassment','disapproval','disappointment','annoyance','anger','nervousness','remorse','grief','disgust'}
# ambiguous = {'realization','surprise','curiosity','confusion','neutral'}

# !pip install -q contractions
import contractions

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

data.drop('id', inplace=True, axis=1)
data.drop('example_very_unclear', inplace=True, axis=1)
data["cleaned_text"] = data["text"].progress_apply(clean_text)
data['emotion'] = (data.iloc[:, 1:] == 1).idxmax(1)
# data = data[ ['cleaned_text', 'emotion'] + [ col for col in data.columns if col not in ['text', 'cleaned_text', 'emotion'] ] ]
data = data[ ['cleaned_text', 'emotion'] ]
data = data[data['cleaned_text'] != '']

train_and_validation, test = train_test_split(data, test_size=0.1, 
                                shuffle=True, random_state=42)
train, validation = train_test_split(train_and_validation, test_size=0.2, shuffle=True, random_state=42)

# train.to_csv(train_dataset_path, index=None)
# validation.to_csv(validation_dataset_path, index=None)
# test.to_csv(test_dataset_path, index=None)

print(train.shape)
print(validation.shape)
print(test.shape)
# 152071  72%   80%
# 38018   18%   20%
# 21122   10%

df_train = train
df_validation = validation
df_test = test

df_train = df_train.rename(columns={'cleaned_text': 'Text', 'emotion': 'Emotion'})
df_validation = df_validation.rename(columns={'cleaned_text': 'Text', 'emotion': 'Emotion'})
df_test = df_test.rename(columns={'cleaned_text': 'Text', 'emotion': 'Emotion'})

##########################################################################


#removing duplicated values
index = df_train[df_train.duplicated() == True].index
df_train.drop(index, axis = 0, inplace = True)
df_train.reset_index(inplace=True, drop = True)

#removing duplicated text 
index = df_train[df_train['Text'].duplicated() == True].index
df_train.drop(index, axis = 0, inplace = True)
df_train.reset_index(inplace=True, drop = True)

index = df_validation[df_validation.duplicated() == True].index
df_validation.drop(index, axis = 0, inplace = True)
df_validation.reset_index(inplace=True, drop = True)

#removing duplicated text 
index = df_validation[df_validation['Text'].duplicated() == True].index
df_validation.drop(index, axis = 0, inplace = True)
df_validation.reset_index(inplace=True, drop = True)

index = df_test[df_test.duplicated() == True].index
df_test.drop(index, axis = 0, inplace = True)
df_test.reset_index(inplace=True, drop = True)

#removing duplicated text 
index = df_test[df_test['Text'].duplicated() == True].index
df_test.drop(index, axis = 0, inplace = True)
df_test.reset_index(inplace=True, drop = True)

##########################################################################
df_val = df_validation

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

nltk.download('wordnet')
stop_words = set(stopwords.words("english"))

if use_nltk_preprocessing:
    df_train= normalize_text(df_train)
    df_test= normalize_text(df_test)
    df_val= normalize_text(df_val)

    # print(df_train.head(20))

##########################################################################

#Splitting the text from the labels
X_train = df_train['Text']
y_train = df_train['Emotion']

X_test = df_test['Text']
y_test = df_test['Emotion']

X_val = df_val['Text']
y_val = df_val['Emotion']

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
y_val = le.transform(y_val)

#Convert the class vector (integers) to binary class matrix
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

# Tokenize words
tokenizer = Tokenizer(oov_token='UNK')
tokenizer.fit_on_texts(pd.concat([X_train, X_test], axis=0))

sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)
sequences_val = tokenizer.texts_to_sequences(X_val)

df_train_maxlen = max([len(t) for t in df_train['Text']])
df_test_maxlen = max([len(t) for t in df_test['Text']])
df_val_maxlen = max([len(t) for t in df_val['Text']])
# print(df_train_maxlen)
# print(df_test_maxlen)
# print(df_val_maxlen)
df_maxlen = max([df_train_maxlen, df_test_maxlen, df_val_maxlen])
print(df_maxlen)

X_train = pad_sequences(sequences_train, maxlen=df_maxlen, truncating=df_truncating)
X_test = pad_sequences(sequences_test, maxlen=df_maxlen, truncating=df_truncating)
X_val = pad_sequences(sequences_val, maxlen=df_maxlen, truncating=df_truncating)

vocabSize = len(tokenizer.index_word) + 1
print(f"Vocabulary size = {vocabSize}")

##########################################################################

# Read GloVE embeddings


num_tokens = vocabSize # 27711
# embedding_dim = 200 #latent factors or features  
hits = 0
misses = 0
embeddings_index = {}
embedding_matrix = np.zeros((num_tokens, embedding_dim))

if use_glove:
# Read word vectors
    with open(path_to_glove_file, encoding="utf8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs
    print("Found %s word vectors." % len(embeddings_index))

    # Assign word vectors to our dictionary/vocabulary
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))

##########################################################################

# Build neural network architecture

adam = Adam(learning_rate=learning_rate)

model = Sequential()
if use_glove:
    model.add(Embedding(vocabSize, embedding_dim, input_length=X_train.shape[1], weights=[embedding_matrix], trainable=False))
else: 
    model.add(Embedding(vocabSize, embedding_dim))
# model.add(Bidirectional(LSTM(256, dropout=0.2,recurrent_dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(lstm_dim, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout)))
# model.add(Bidirectional(LSTM(128, dropout=0.2,recurrent_dropout=0.2, return_sequences=True)))
# model.add(Bidirectional(LSTM(128, dropout=0.2,recurrent_dropout=0.2)))
model.add(Dense(28, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()

if importing_model:
    import tensorflow as tf
    model = tf.keras.models.load_model(imported_model_path)
    # model = tf.keras.models.load_model(imported_model_path, custom_objects=adam)
else:
    history = model.fit(X_train,
                        y_train,
                        validation_data=(X_val, y_val),
                        verbose=1,
                        batch_size=training_batch_size,
                        epochs=training_epoch,
                    )
    
    model.save(exported_model_path_h5)
    model.save(exported_model_path)

#print the overall loss and accuracy
model.evaluate(X_val, y_val, verbose=1) 
#print the overall loss and accuracy
model.evaluate(X_test, y_test, verbose=1) 

predicted = model.predict(X_test)
y_pred = predicted.argmax(axis=-1)
print(classification_report(le.transform(df_test['Emotion']), y_pred))

# Classify custom sample

for sentence in sentences:
    print(sentence)
    sentence = normalized_sentence(sentence)
    sentence = tokenizer.texts_to_sequences([sentence])
    sentence = pad_sequences(sentence, maxlen=df_maxlen, truncating=df_truncating)
    
    result = model.predict(sentence)[0]
    best_three_indices = np.argsort(result)[::-1][:3]
    label_best_three = le.inverse_transform(best_three_indices)
    result_best_three = result[best_three_indices]
    for i in range(3):
        print(f"{label_best_three[i]}: {'{:.4%}'.format(result_best_three[i])}")
    print()


import pickle
# with open('tokenizer.pickle', 'wb') as handle:
#     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open("labelencoder.pickle", "wb") as f:
#     pickle.dump(le, f)

# admiration: 85.6613%
# neutral: 5.2887%
# approval: 2.5901%
