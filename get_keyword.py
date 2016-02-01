import re
import nltk
from nltk.corpus import stopwords 
import nltk.data
import gensim, logging
from gensim.models import word2vec
from gensim.models import Word2Vec
import numpy 
import sklearn
from sklearn.metrics import pairwise_distances
from numpy import argsort, array
import pandas as pd
from numpy import *
from collections import Counter

with open("dream_luther.txt") as file: 
    text_review = file.read()

model = word2vec.Word2Vec.load("w2v")
num_features = 100

def text_to_wordlist( text, remove_stopwords=True):

    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", text)

    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return(meaningful_words)

def get_feature_vec(words, model, num_features):
    # Function to get feature vec of words and clean stopwords from model' vocabulary

    index2word_set = set(model.index2word)
    # Loop over each word in the text and, if it is in the model's
    # vocabulary, add its feature vector to the total
    clean_text = []
    for word in words:
        if word in index2word_set: 
            clean_text.append(model[word])

    return clean_text

clean_train_text = (text_to_wordlist( text_review, \
        remove_stopwords=True ))


trainDataVecs = get_feature_vec (clean_train_text, model, num_features )
trainData = numpy.asarray(trainDataVecs)

#calculate cosine similarity matrix to use in pagerank algorithm for dense matrix, it is not
#fast for sparse matrix
cosine = 1-pairwise_distances(trainData, metric="cosine")


#pagerank powermethod 
def powerMethod(A,x0,m,iter):
    n = A.shape[1]
    delta = m*(array([1]*n,dtype='float64')/n) 
    for i in range(iter):
        x0 = dot((1-m),dot(A,x0)) + delta
    return x0


n = cosine.shape[1] # A is n x n
m = 0.15
x0 = [1]*n

pagerank_values = powerMethod(cosine,x0,m,130)
srt = numpy.argsort(pagerank_values)

a = srt[0:20]
    
keywords_list = []
for words in a:
    keywords_list.append(clean_train_text[words])

print keywords_list
