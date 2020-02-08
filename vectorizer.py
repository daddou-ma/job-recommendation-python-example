import pandas as pd
import pickle


from sklearn.feature_extraction.text import TfidfVectorizer

class Vectorizer:
  def __init__(self, vectorizer=None):
    if vectorizer is not None:
      self.vectorizer = vectorizer
    else:
      self.vectorizer = TfidfVectorizer()

  @classmethod
  def load(cls, filename):
    return cls(vectorizer=pickle.load(open(filename, "rb")))

  def fit_transform(self, texts):
    return self.vectorizer.fit_transform((texts))

  def transform(self, texts):
    return self.vectorizer.transform((texts))

  def save_vectorizer(self, filename):
    pickle.dump(self.vectorizer, open(filename,"wb"))

  def save_tfidfs(self, tfidfs, filename):
    pickle.dump(tfidfs, open(filename,"wb"))

  def save_tfidfs_as_csv(self, tfidfs, filename):
    vectors = pd.DataFrame(tfidfs.toarray(), columns=self.vectorizer.get_feature_names())
    vectors.to_csv(filename, index = False)
