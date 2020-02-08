import pandas as pd
import numpy as np
import pickle
import json
from vectorizer import Vectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# Load Raw Data Files
df_users = pd.read_csv("./data/users.csv")
df_jobs = pd.read_csv("./raw/Combined_Jobs_Final.csv")

u = 326
index = np.where(df_users['applicant_id'] == u)[0][0]
user_query = df_users.iloc[[index]]

vectorizer = Vectorizer.load('./pickles/vectorizer.pkl')

tfidf_jobs = pickle.load(open("./pickles/tfidf_jobs.pkl", "rb"))

tfidf_user = vectorizer.transform((user_query['text']))

cos_similarity_tfidf = list(map(lambda x: cosine_similarity(tfidf_user, x), tfidf_jobs))

def get_recommendation(top, df_all, scores):
  recommendation = pd.DataFrame(columns = ['ApplicantID', 'JobID',  'title', 'job_description', 'score'])
  count = 0
  for i in top:
      recommendation.at[count, 'ApplicantID'] = u
      recommendation.at[count, 'JobID'] = df_all['Job.ID'][i]
      recommendation.at[count, 'title'] = df_all['Title'][i]
      recommendation.at[count, 'job_description'] = df_all['Job.Description'][i]
      recommendation.at[count, 'score'] =  scores[count]
      count += 1
  return recommendation

top = sorted(range(len(cos_similarity_tfidf)), key=lambda i: cos_similarity_tfidf[i], reverse=True)[:10]
list_scores = [cos_similarity_tfidf[i][0][0] for i in top]

recommendations = get_recommendation(top,df_jobs, list_scores)
recommendations.to_csv('./vectors/recommanded.csv', index = False)
print(recommendations)