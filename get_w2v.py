import re
import nltk
import nltk.data
from gensim.models import word2vec


with open("Wikipedia_Text.txt") as file:
    text_review = file.read()

# Function to convert a document to a sequence of words,
def text_to_wordlist(text, remove_stopwords=False):
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]", " ", text)

    # 3. Convert words to lower case and split them
    words = review_text.lower().split()

    return (words)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Define a function to split a text into parsed sentences
def text_to_sentences(text, tokenizer, remove_stopwords=False):
    # Function to split a text into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    review = text.decode('utf-8').replace(u'\u014c\u0106\u014d', '-')
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append(text_to_wordlist(sentence, remove_stopwords=False))

    return sentences

sentences = []

# Parsing sentences from training set
for review in text_review:
    sentences += text_to_sentences(text_review, tokenizer, remove_stopwords=False)

num_features = 100
model = word2vec.Word2Vec(sentences, size=num_features, window=5, min_count=0)
model_name = "w2v"
model.save(model_name)
# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)
model = word2vec.Word2Vec.load(model_name)

print type(model.syn0)
print model.syn0.shape
# it gives vector representation of a word
print model["and"]

# bigram_transformer = gensim.models.Phrases(sentences)
# model = word2vec.Word2Vec(bigram_transformed[sentences], size=100, ...)

print model.similarity('word_1','word_2')
