import pandas as pd
import numpy as np
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

#initializing tfidf vectorizer
vectorizer = TfidfVectorizer(max_features = 1024)

# Load Raw Data Files
df_jobs = pd.read_csv("./clean/jobs.csv")

tfidf_jobs = vectorizer.fit_transform((df_jobs['text'])) #fitting and transforming the vector

pickle.dump(vectorizer.vocabulary_,open("feature.pkl","wb"))

vectors = pd.DataFrame(tfidf_jobs.toarray(), columns=vectorizer.get_feature_names())

vectors.to_csv('./vectors/jobs.csv')

