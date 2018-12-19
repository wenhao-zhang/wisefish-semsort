# https://www.kaggle.com/ngyptr/lstm-sentiment-analysis-keras - algorithm

import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import csv

data = pd.read_csv('D:\\SFU\\cmpt-413\\nlpclass-1187-g-wisefish\\project\\yelpStarModel\\sentiment_yelp_data_ngram_split_10000_sample.csv')

max_features = 3000
print("Preprocessing Text Sequences ... ")
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data['Text'].values)
text_sequences = tokenizer.texts_to_sequences(data['Text'].values)
text_sequences = pad_sequences(text_sequences)

print("Creating Model ...")
model = Sequential()
model.add(Embedding(max_features, 128, input_length=text_sequences.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

sentiments = pd.get_dummies(data['Sentiment']).values

text_sequences_train, text_sequence_test, sentiments_train, sentiments_test = train_test_split(text_sequences,sentiments, test_size = 0.20, random_state = 42)
print(text_sequences_train.shape,sentiments_train.shape)
print(text_sequence_test.shape,sentiments_test.shape)

batch_size = 64
model.fit(text_sequences_train, sentiments_train, epochs = 5, batch_size=batch_size, verbose = 1)

validation_size = 1000

sentiments_validate = sentiments_test[-validation_size:]
text_sequence_validate = text_sequence_test[-validation_size:]
sentiments_test = sentiments_test[:-validation_size]
text_sequence_test = text_sequence_test[:-validation_size]

positive_count, positive_correct = 0, 0,
negative_count, negative_correct = 0, 0
for x in range(len(text_sequence_validate)):
    
    result = model.predict(text_sequence_validate[x].reshape(1,text_sequence_test.shape[1]),batch_size=1,verbose = 2)[0]
    
    if np.argmax(result) == np.argmax(sentiments_validate[x]):
        if np.argmax(sentiments_validate[x]) == 0:
            negative_correct += 1
        else:
            positive_correct += 1
       
    if np.argmax(sentiments_validate[x]) == 0:
        negative_count += 1
    else:
        positive_count += 1

print("Positive Accuracy: ", positive_correct/positive_count*100, "%")
print("Negative Accuracy: ", negative_correct/negative_count*100, "%")

model.save("D:\\SFU\\cmpt-413\\nlpclass-1187-g-wisefish\\project\\yelpStarModel\\nltk_split_sentence_model.h5")
print('Done.')