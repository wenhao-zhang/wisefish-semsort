from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import numpy as np
import pandas as pd
model = load_model("D:\\SFU\\cmpt-413\\nlpclass-1187-g-wisefish\\project\\yelpStarModel\\nltk_split_sentence_model.h5")
data = pd.read_csv('D:\\SFU\\cmpt-413\\nlpclass-1187-g-wisefish\\project\\yelpStarModel\\sentiment_yelp_data_20000_sample.csv')

max_features = 3000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data['Text'].values)

sentences = [
    ["Why can't we be friends!"],
    ["5/5 for ambiance, good music, and awesome vegan options."],
    ["Graeme, Mitch, Jonathan, Wenhao are the wisefish team"],
    ["What is this Hipster FUCKERY"],
    ["Horrible quality of interior ."],
    ["Solid, high quality, comfortable and quiet ."],
    ["Toyota's quality is slipping ."],
]

for sentence in sentences:
    #vectorizing the tweet by the pre-fitted tokenizer instance
    tokenized_sentence = tokenizer.texts_to_sequences(sentence)
    #padding the tweet to have exactly the same shape as `embedding_2` input
    tokenized_sentence = pad_sequences(tokenized_sentence, maxlen=975, dtype='int32', value=0)
    # print(sentence)
    sentiment = model.predict(tokenized_sentence,batch_size=1,verbose = 2)[0]
    print(sentiment)
    if(np.argmax(sentiment) == 0):
        print(f'{sentence[0]} -> is negative')
    elif (np.argmax(sentiment) == 1):
        print(f'{sentence[0]} -> is positive')