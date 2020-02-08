import pandas as pd
import pickle
from vectorizer import Vectorizer

# Load Raw Data Files

def vectorize_jobs(df_jobs, vectorizer_path, tfidfs_path, tfidfs_csv_path, debug=False):
  if debug:
    print('[Job Vectorization 2/7] Transforming Data \n')

  # df_jobs["text"] =  df_jobs["position"].map(str) + " " + df_jobs["company"].map(str) + " " + df_jobs["city"].map(str) + " " + df_jobs['employment_type'].map(str) + " " + df_jobs['job_description'].map(str) + " " + df_jobs['title'].map(str)
  # df_jobs['text'] = df_jobs['text'].apply(clean_txt)
  # df_jobs = df_jobs[['job_id', 'text']]

  #initializing tfidf vectorizer
  if debug:
    print('[Job Vectorization 3/7] Initializing Vectorizer \n')
  vectorizer = Vectorizer()

  if debug:
    print('[Job Vectorization 4/7] Tranforming/Vectorizing data \n')
  tfidf_jobs = vectorizer.fit_transform((df_jobs['text'])) #fitting and transforming the vector

  if debug:
    print('[Job Vectorization 5/7] saving vectorizer to {path} \n'.format(path=vectorizer_path))
  vectorizer.save_vectorizer(vectorizer_path)

  if debug:
    print('[Job Vectorization 6/7] saving tfidf to {path} \n'.format(path=tfidfs_path))
  vectorizer.save_tfidfs(tfidf_jobs, tfidfs_path)

  if debug:
    print('[Job Vectorization 7/7] saving tfidf to {path} \n'.format(path=tfidfs_csv_path))
  #vectorizer.save_tfidfs_as_csv(tfidf_jobs, tfidfs_csv_path)


def execute_vectorize_jobs(debug=False):
  if debug:
    print('[Job Vectorization 1/7] Loading data From "./data/jobs.csv" \n')

  data = pd.read_csv("./data/jobs.csv")
  vectorize_jobs(
    data,
    vectorizer_path='./pickles/vectorizer.pkl',
    tfidfs_path='./pickles/tfidf_jobs.pkl',
    tfidfs_csv_path='./vectors/tfidf_jobs.csv',
    debug=debug
  )

execute_vectorize_jobs(debug=True)