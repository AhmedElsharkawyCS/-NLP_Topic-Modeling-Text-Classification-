#(form the lecture )we have a two approaches exist to convert text into the corresponding numerical form. The Bag of Words Model and the Word Embedding Model.
#we will use the bag of word technique(note:all unique words are converted into features)
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
class text_to_vectorizer:
    def _vectorizer(data):
          vectorizer=CountVectorizer(max_features=2000,min_df=5,max_df=0.7,stop_words=stopwords.words('english'))
          #converts text documents into corresponding numeric features
          x=vectorizer.fit_transform(data)
#          print(vectorizer.get_feature_names())
          vectors=x.toarray()
          return vectors
          