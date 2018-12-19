import pandas as pd 
from nltk.tokenize import sent_tokenize
import random 
import csv

def basicDataFormatting():
    dataframe = pd.read_csv('D:\\SFU\\cmpt-413\\nlpclass-1187-g-wisefish\\project\\yelpStarModel\\reviews.csv')
    sample_size = 10000
    neg_stars = [1,2]
    pos_stars = [4,5]

    pos = dataframe.loc[dataframe['stars'].isin(pos_stars)]
    neg = dataframe.loc[dataframe['stars'].isin(neg_stars)]

    pos_sample = pos.sample(sample_size)
    neg_sample = neg.sample(sample_size)

    print('Positive: ', pos_sample.size)
    print('Negative: ', neg_sample.size)

    sampled_data = []

    for i, row in pos_sample.iterrows():
        sampled_data.append(['Positive', row["text"]])
    for i, row in neg_sample.iterrows():
        sampled_data.append(['Positive', row["text"]])

    random.shuffle(sampled_data)
    outframe = pd.DataFrame(sampled_data, columns=['Sentiment', 'Text'])

    outframe.to_csv(f'D:\\SFU\\cmpt-413\\nlpclass-1187-g-wisefish\\project\\yelpStarModel\\sentiment_yelp_data_split_{sample_size*2}_sample.csv')
    print('Done')
    return


def sentenceSplitDataFormatting():
    dataframe = pd.read_csv('D:\\SFU\\cmpt-413\\nlpclass-1187-g-wisefish\\project\\yelpStarModel\\reviews.csv')
    sample_size = 5000
    neg_stars = [1,2]
    pos_stars = [4,5]

    pos = dataframe.loc[dataframe['stars'].isin(pos_stars)]
    neg = dataframe.loc[dataframe['stars'].isin(neg_stars)]

    pos_sample = pos.sample(sample_size)
    neg_sample = neg.sample(sample_size)

    print('Positive: ', pos_sample.size)
    print('Negative: ', neg_sample.size)

    sampled_data = []

    for _, row in pos_sample.iterrows():
        review = row["text"]
        tokenized_review = sent_tokenize(review)
        for sentence in tokenized_review:
            sampled_data.append(['Positive', sentence])
    for _, row in neg_sample.iterrows():
        review = row["text"]
        tokenized_review = sent_tokenize(review)
        for sentence in tokenized_review:
            sampled_data.append(['Negative', sentence])

    random.shuffle(sampled_data)
    outframe = pd.DataFrame(sampled_data, columns=['Sentiment', 'Text'])

    outframe.to_csv(f'D:\\SFU\\cmpt-413\\nlpclass-1187-g-wisefish\\project\\yelpStarModel\\sentiment_yelp_data_sentence_split_{sample_size*2}_sample.csv')
    print('Done')


    return


def generate_ngrams(sentence, n):
    
    if n >= len(sentence):
        return sentence

    words = sentence.split(" ")
    ngrams = []
    
    for i in range(len(words)-n):
        ngram = ""
        for j in range(n):
            ngram = f'{ngram}{" " + words[i+j]}'    
        ngrams.append(ngram)
    words = words + ngrams
    return ' '.join(words)

def ngramDataFormatting():
    dataframe = pd.read_csv('D:\\SFU\\cmpt-413\\nlpclass-1187-g-wisefish\\project\\yelpStarModel\\reviews.csv')
    sample_size = 5000
    neg_stars = [1,2]
    pos_stars = [4,5]

    pos = dataframe.loc[dataframe['stars'].isin(pos_stars)]
    neg = dataframe.loc[dataframe['stars'].isin(neg_stars)]

    pos_sample = pos.sample(sample_size)
    neg_sample = neg.sample(sample_size)

    print('Positive: ', pos_sample.size)
    print('Negative: ', neg_sample.size)

    sampled_data = []

    for i, row in pos_sample.iterrows():
        review = row["text"]
        tokenized_review = sent_tokenize(review)
        for sentence in tokenized_review:
            ngram_sentence = generate_ngrams(sentence, 2)
            sampled_data.append(['Positive', ngram_sentence])
    for i, row in neg_sample.iterrows():
        review = row["text"]
        tokenized_review = sent_tokenize(review)
        for sentence in tokenized_review:
            ngram_sentence = generate_ngrams(sentence, 2)
            sampled_data.append(['Negative', ngram_sentence])

    random.shuffle(sampled_data)
    outframe = pd.DataFrame(sampled_data, columns=['Sentiment', 'Text'])

    outframe.to_csv(f'D:\\SFU\\cmpt-413\\nlpclass-1187-g-wisefish\\project\\yelpStarModel\\sentiment_yelp_data_ngram_split_{sample_size*2}_sample.csv')
    print('Done')

    return

ngramDataFormatting()