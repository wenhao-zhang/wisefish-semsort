from os.path import abspath
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import numpy as np
import pandas as pd
import progressbar
# model 4
# pos_acc 88.30645161290323 %
# neg_acc 96.62698412698413 %
def basicSentimentSort(sentences, with_sentiment=False):
    #setup
    model = load_model(abspath('./yelpModel/model5.h5'))
    # most common words used as features
    max_features = 3000
    # Tokenizer for preprocessing sentence to wordVecs
    tokenizer = Tokenizer(num_words=max_features, split=' ')
    print("Sorting: ")
    bar = progressbar.ProgressBar(max_value=len(sentences))

    sorted_sentences = {"Positive":[], "Negative":[]}

    for i, sentence in enumerate(sentences):
        # vectorize the sentence
        tokenized_sentence = tokenizer.texts_to_sequences(sentence)
        # pad the sentences for the network
        tokenized_sentence = pad_sequences(tokenized_sentence, maxlen=944, dtype='int32', value=0)
        # generate the sentiment prediction from the model -> 2 element list [negative <float> , positive <float>]
        sentiment = model.predict(tokenized_sentence,batch_size=1,verbose = 2)[0]
        #place sentence in the output dict
        if sentiment[1] > sentiment[0]:
            if with_sentiment:
                sorted_sentences["Positive"].append((sentence, sentiment))
            else:
                sorted_sentences["Positive"].append(sentence)
        else:
            if with_sentiment:
                sorted_sentences["Negative"].append((sentence, sentiment))
            else:
                sorted_sentences["Negative"].append(sentence)
        bar.update(i)
    return sorted_sentences

def voteSentimentSort(sentences):
    # good at finding positive sentiment
    model1 = load_model(abspath('./yelpModel/model1.h5'))
    # good at finding negative sentiment
    model2 = load_model(abspath('./yelpModel/model2.h5'))
    model3 = load_model(abspath('./yelpModel/model5.h5'))
    max_features = 3000
    tokenizer = Tokenizer(num_words=max_features, split=' ')

    sentence_sentiment = {}
    print("\nPredict with Model 1: ")
    bar1 = progressbar.ProgressBar(max_value=len(sentences))
    for i, sentence in enumerate(sentences):
        #vectorizing
        tokenized_sentence = tokenizer.texts_to_sequences(sentence)
        #padding
        tokenized_sentence = pad_sequences(tokenized_sentence, maxlen=975, dtype='int32', value=0)
        sentiment = model1.predict(tokenized_sentence,batch_size=1,verbose = 2)[0]
        sentence_sentiment[sentence] = [sentiment[0], sentiment[1]]
        bar1.update(i)
    print("\nPredict With Model 2: ")
    bar2 = progressbar.ProgressBar(max_value=len(sentences))
    for i, sentence in enumerate(sentences):
        #vectorizing
        tokenized_sentence = tokenizer.texts_to_sequences(sentence)
        #padding
        tokenized_sentence = pad_sequences(tokenized_sentence, maxlen=273, dtype='int32', value=0)
        sentiment = model2.predict(tokenized_sentence,batch_size=1,verbose = 2)[0]
        sentence_sentiment[sentence].append(sentiment[0])
        sentence_sentiment[sentence].append(sentiment[1])
        bar2.update(i)

    print("\nPredict With Model 3: ")
    bar3 = progressbar.ProgressBar(max_value=len(sentences))
    for i, sentence in enumerate(sentences):
        #vectorizing
        tokenized_sentence = tokenizer.texts_to_sequences(sentence)
        #padding
        tokenized_sentence = pad_sequences(tokenized_sentence, maxlen=946, dtype='int32', value=0)
        sentiment = model3.predict(tokenized_sentence,batch_size=1,verbose = 2)[0]
        sentence_sentiment[sentence].append(sentiment[0])
        sentence_sentiment[sentence].append(sentiment[1])
        bar3.update(i)

    print("\nSorting: ")
    sorted_sentences = {"Positive":[], "Negative":[]}
    for sentence, sentiment in sentence_sentiment.items():
        print(sentiment)
        m1_neg = sentiment[0]
        m1_pos = sentiment[1]
        m2_neg = sentiment[2]
        m2_pos = sentiment[3]
        m3_neg = sentiment[4]
        m3_pos = sentiment[5]

        avg_pos =  (m1_pos + m2_pos + m3_pos) / 3
        avg_neg =  (m1_neg + m2_neg + m3_neg) / 3

        if avg_neg < avg_pos:
            sorted_sentences["Positive"].append(sentence)
        else:
            sorted_sentences["Negative"].append(sentence)
    return sorted_sentences
