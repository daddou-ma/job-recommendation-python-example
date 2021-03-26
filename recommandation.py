import pandas as pd
import numpy as np
import pickle
import json
from vectorizer import Vectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from config import get_raw_file_path, get_data_file_path, get_pickle_file_path, get_result_file_path
import sys

if len(sys.argv) == 1:
  sys.exit()

# Load Raw Data Files
df_users = pd.read_csv(get_data_file_path('users'))
df_jobs = pd.read_csv(get_raw_file_path('jobs'))

u = int(sys.argv[1]) #326
index = np.where(df_users['applicant_id'] == u)[0][0]
user_query = df_users.iloc[[index]]

print('[User Information -------------------------------]')
print(df_users[df_users['applicant_id'] == u])

vectorizer = Vectorizer.load(get_pickle_file_path('vectorizer'))

tfidf_jobs = pickle.load(open(get_pickle_file_path('tfidf'), "rb"))

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
recommendations.to_csv(get_result_file_path('output'), index = False)
print('[Recommendations -------------------------------]')
print(recommendations)