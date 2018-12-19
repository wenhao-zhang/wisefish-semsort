import sys
from SemSort import SemSort
import optparse, json
import pandas as pd

import nltk
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')

SUMMARY_LINES = 10

def parseReviews():
  # Load all reviews and keep only three columns
  reviewsChunked = pd.read_json('yelp_academic_dataset_review.json', lines=True, convert_dates=False, dtype=False, chunksize=500)
  reviews = pd.concat([ chunk for chunk in reviewsChunked ])
  return reviews[['business_id', 'stars', 'text']]

def parseBusinesses(city, state):
  # Read all businesses and keep only in specified sities
  businessesChunked = pd.read_json('yelp_academic_dataset_business.json', lines=True, convert_dates=False, dtype=False, chunksize=500)
  businesses = pd.concat([ chunk for chunk in businessesChunked ])
  return businesses[(businesses['city'] == city) & (businesses['state'] == state)][['business_id', 'stars', 'name', 'review_count']]

reviewsBusinessDF = None

# For concurrency, run with chunks
def runOnBusinesses(args):
  chunkNum, businesses = args
  with open(f"results_chunk_{chunkNum}.json", 'w') as f:
    for _, business in businesses.iterrows():
      result = runOnBusiness(business)
      f.write(json.dumps(result) + "\n")

def runOnBusiness(business):
  business_id = business['business_id']
  reviews = reviewsBusinessDF[reviewsBusinessDF['business_id'] == business_id][['stars', 'text']]
  reviews.columns = ['Stars', 'Text']
  result = {
      'business_id': business_id,
      'business_name': business['name'],
      'review_count': len(reviews),
      'stars': business['stars'],
      # Run the SemSort on this business' reviews.
      'result': SemSort(reviews, multipath=False, num_bullet_points=SUMMARY_LINES)
  }
  print(json.dumps(result, indent=4))
  return result

from multiprocessing import Pool

if __name__ == "__main__":
  businesses = parseBusinesses('Toronto', 'ON')
  reviewsBusinessDF = parseReviews()

  # RUN WITH BELOW FOR SERIAL
  with open(f"results.json", 'w') as f:
    for _, business in businesses.iterrows():
      result = runOnBusiness(business)
      f.write(json.dumps(result) + "\n")

  # RUN WITH BELOW FOR CONCURRENT
  # chunksize = 2000
  # chunks = [(i, businesses[i:i + chunksize]) for i in range(0, len(businesses), chunksize)]
  # p = Pool(3)
  # p.map(runOnBusinesses, chunks)
  # p.close()
  # p.join()
