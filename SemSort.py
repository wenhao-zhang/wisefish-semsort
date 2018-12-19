from sum_basic import SumBasic
from yelpModel.sentimentSort import basicSentimentSort,voteSentimentSort
import pandas as pd

def pickTop(n, reviews, idx):
    sorted_reviews = sorted(reviews,  key=lambda tup: tup[1][idx],reverse=True)
    top = []
    for i in range(n):
        if i < len(sorted_reviews):            
            top.append(sorted_reviews[i][0])
    return top

def SemSort(data, multipath=False, num_bullet_points=4):
    sorted = {"Positive":[], "Negative":[]}
    if multipath: 
        #Split Reviews by star rating
        oneStar = []
        twoStar = []
        fourStar = []
        fiveStar = []
        for _, row in data.iterrows():
            if row["Stars"] == 1:
                oneStar.append(row["Text"])
            elif row["Stars"] == 2:
                twoStar.append(row["Text"])
            elif row["Stars"] == 4:
                fourStar.append(row["Text"])
            elif row["Stars"] == 5:
                fiveStar.append(row["Text"])
        
        low_reviews = oneStar + twoStar
        high_reviews = fiveStar + fourStar
        
        #Summarize low start reviews
        neg_summarizer = SumBasic(low_reviews)
        neg_summary = neg_summarizer.get_summary(10)
        # Sort low star reviews
        neg_sorted = basicSentimentSort(neg_summary, with_sentiment=True)
        #Summarize high star reviews
        pos_summarizer = SumBasic(high_reviews)
        pos_summary = pos_summarizer.get_summary(10)
        #Sort high star reviews
        pos_sorted = basicSentimentSort(pos_summary, with_sentiment=True)

        total_pos = pos_sorted["Positive"] + neg_sorted["Positive"]
        total_neg = pos_sorted["Negative"] + neg_sorted["Negative"]
        
        #Merge the reviews by taking the top n from the positive and negative sentences
        sorted["Positive"] = pickTop(num_bullet_points, total_pos, 1)
        sorted["Negative"] = pickTop(num_bullet_points, total_neg, 0)
            

    else:
        # collect reviews into list of strings
        data_to_be_sorted = []
        for _, row in data.iterrows():
            data_to_be_sorted.append(row["Text"])

        #generate summaries of the reviews
        summarizer = SumBasic(data_to_be_sorted)
        summary = summarizer.get_summary(num_bullet_points)
        #sort summaries with RNN preditctions
        sorted = basicSentimentSort(summary)

    return sorted

if __name__ == "__main__":
    csv_data = pd.read_csv(abspath('./yelpModel/CF_Toronto_Eaton_Centre_reviews.csv'))
    sorted = SemSort(csv_data, multipath=False, num_bullet_points=10)
    print("\nPositive")
    for i, sentence in enumerate(sorted["Positive"]):
        print(f'{i+1}: {sentence}')
    print("\nNegative")
    for i, sentence in enumerate(sorted["Negative"]):
        print(f'{i+1}: {sentence}')
