from __future__ import print_function
from os import listdir
from os.path import isfile, join
import numpy as np
import sys
import json
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.externals import joblib




stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")

onlyfiles = [f for f in listdir(str(sys.argv[1])) if isfile(join(str(sys.argv[1]), f))]
#print(onlyfiles)
titles=[]
synopses=[]
patt = re.compile('(\s*)tablespoon|tablespoons|teaspoon|freshly|fresh|chopped|cut|ounce|pounds|sliced|finely|large|s|cup(\s*)')
n = len(onlyfiles)
j=0
for i in range(0,n):
        with open(os.path.abspath(os.path.join(sys.argv[1],onlyfiles[i])), encoding="utf8") as file:
             data = json.load(file)
             #print(i)
             titles.append(data['id'])
             sent = (''.join(data["ingredientLines"])).replace(',', '')
             sent = patt.sub('',sent)
             synopses.append(sent)


def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


#not super pythonic, no, not at all.
#use extend so it's a big flat list of vocab
totalvocab_stemmed = []
totalvocab_tokenized = []
for i in synopses:
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)


vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(synopses) #fit the vectorizer to synopses

print(tfidf_matrix.shape)

terms = tfidf_vectorizer.get_feature_names()

dist = 1 - cosine_similarity(tfidf_matrix)
#print(dist)

num_clusters = 8

km = KMeans(n_clusters=num_clusters)

km.fit(tfidf_matrix)

clusters = km.labels_.tolist()

#uncomment the below to save your model 
#since I've already run my model I am loading from the pickle

joblib.dump(km,  'doc_cluster.pkl')

km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()

films = { 'title': titles, 'synopsis': synopses, 'cluster': clusters }

frame = pd.DataFrame(films, index = [clusters] , columns = ['title', 'cluster'])

frame['cluster'].value_counts() #number of films per cluster (clusters from 0 to 4)

#grouped = frame['rank'].groupby(frame['cluster']) #groupby cluster for aggregation purposes

#grouped.mean() #average rank (1 to 100) per cluster


print("Top terms per cluster:")
print()
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    
    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print() #add whitespace
    print() #add whitespace
    
    print("Cluster %d titles:" % i, end='')
    for title in frame.ix[i]['title'].values.tolist():
        print(' %s,' % title, end='')
    print() #add whitespace
    print() #add whitespace


tfidf_matrix_test = tfidf_vectorizer.transform(["2 tablespoons coconut oil or organic canola oil ¼ teaspoon turmeric 1 tablespoon cumin seeds 1 large onion sliced 5 garlic cloves minced 2-inch knob ginger peeled and grated/minced ½ teaspoon ground cinnamon ½ teaspoon ground cardamom ¼ teaspoon ground cloves ¼-1/2 teaspoon cayenne pepper ¾ teaspoon sea salt plus more to taste 1 cup peeled and sliced carrots 2 cups ½-inch cubed peeled potatoes (from about 2 medium potatoes) 1 (13.5-ounce) can coconut milk 1 cup water 1 red bell pepper seeded and sliced ⅓ cup peanut butter 2 tablespoons natural cane sugar 1 (14-ounce) package extra-firm tofu "])

prediction = km.predict(tfidf_matrix_test)
print(prediction)