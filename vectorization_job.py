import pandas as pd
import pickle
from vectorizer import Vectorizer
from config import get_data_file_path, get_pickle_file_path

def vectorize_jobs(df_jobs, vectorizer_path, tfidfs_path, debug=False):
  #initializing tfidf vectorizer
  if debug:
    print('[Job Vectorization 2/5] Initializing Vectorizer \n')
  vectorizer = Vectorizer()

  if debug:
    print('[Job Vectorization 3/5] Tranforming/Vectorizing data \n')
  tfidf_jobs = vectorizer.fit_transform((df_jobs['text'])) #fitting and transforming the vector

  if debug:
    print('[Job Vectorization 4/5] saving vectorizer to {path} \n'.format(path=vectorizer_path))
  vectorizer.save_vectorizer(vectorizer_path)

  if debug:
    print('[Job Vectorization 5/5] saving tfidf to {path} \n'.format(path=tfidfs_path))
  vectorizer.save_tfidfs(tfidf_jobs, tfidfs_path)



def execute_vectorize_jobs(debug=False):
  if debug:
    print('[Job Vectorization 1/5] Loading data From "./data/jobs.csv" \n')

  data = pd.read_csv(get_data_file_path('jobs'))
  vectorize_jobs(
    data,
    vectorizer_path=get_pickle_file_path('vectorizer'),
    tfidfs_path=get_pickle_file_path('tfidf'),
    debug=debug
  )

execute_vectorize_jobs(debug=True)