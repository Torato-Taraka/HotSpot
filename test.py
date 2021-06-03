import pandas as pd
from ast import literal_eval
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import pre
import numpy as np
import process
from hotspot import HotSpot
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
import datetime
import unicodedata
import re
import json
"""
values = [['4.1', '0', '1'], ['5', '2', '3']]
coefs = np.asarray(values, dtype = np.float32)
np.savetxt("test.txt", coefs, fmt='%f')

a = 'hello.csv'
a = a.replace('.csv', '.txt')
print(a)


vector = np.loadtxt('test.txt', dtype=np.float32)
print(vector)

b = coefs
print(np.vstack((coefs, b)))
print(b)

print(type(coefs[0][0]))
"""
"""
words = pre.dataframe_list_to_list(data.Words)
model = Word2Vec(words, sg=1, min_count=0, size=300)
print(model['china'])

model.save('local.model')

data = pd.read_csv('data/2021-5-23.csv')
words = pre.dataframe_list_to_list(data.Words)

model = Word2Vec.load('local.model')
model.build_vocab(words, update=True)
model.train(words, total_examples=model.corpus_count, epochs=model.epochs)
print(model['china'])
"""

"""
text, data = pre.load_json_data_from_dir('test')
print('this is the text:')
print(pre.json_print(text[0]))
print('\nthis is the data:')
print(data)
print(data.info())
"""
"""
data = pd.read_csv('data/2021-5-28.csv')
vector = np.loadtxt('data/2021-5-28.txt', dtype=np.float32)
data.dropna(axis=0, inplace=True)
#data['Summary'] = 0
print(data.info())
"""

data = "asdjflajsdlf的空间里加速度fkdjlsjdklf"
print(data)

data = re.sub(u"[\u4e00-\u9fa5]", "", data)
print(data)

"""
keywords = pre.dataframe_list_to_list(data.Keywords)
keywords = [x for y in keywords for x in y]
print(keywords)
print('=' * 80)

keywords_dict = {}
for tur in keywords:
    print(tur)
    if tur[0] not in keywords_dict:
        keywords_dict.setdefault(tur[0], []).append(tur[1])
    else:
        keywords_dict[tur[0]].append(tur[1])
            
print('=' * 80)
print(keywords_dict)

print('=' * 80)
keywords_cloud = {}
for key, value in keywords_dict.items():
    score = sum(value) / len(value) * len(value)
    keywords_cloud[key] = score
keywords_cloud = list(zip(keywords_cloud.keys(), keywords_cloud.values()))
print(keywords_cloud)
    """    
"""
a = {'x': 0.1}
a.setdefault('y', []).append(0.6)
print(a)
"""
"""
data['Cluster'] = 0
data['Summary'] = 'test'
pre.build_knowledge_graph(data, 1, today)

print(data.loc[:, ['Headline', 'URL']])
hotscore = [90.123]
pre.build_topic(data, 1, today, hotscore)
"""
"""
x = json.load(open('topic/2021-05-26-0.json', 'r'))
pre.json_print(x)
"""
"""
similarity = cosine_similarity(vector)
print(similarity)
cluster = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', n_neighbors=5).fit_predict(similarity)
print(cluster)

print('Hello' in data.columns)

a = pd.DataFrame([[1, 2, 3], [1, 2, 4]])
b = pd.DataFrame([[4, 5], [4, 5]])
c = pd.concat([a, b], ignore_index=True)
print(c)

x = max(set(list(data.Source.values)), key = list(data.Source.values).count)



#hotspot = HotSpot(data, vector)
print()
#print(hotspot.summary)
"""
"""
stopwords = open('stopwords.txt', 'r').read().split('\n')
print(type(stopwords))
print('my' in stopwords)
"""