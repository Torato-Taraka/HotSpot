import json
import os
from rake_nltk import Rake
import networkx
# 一个图结构的相关操作包，没用过无所谓，有兴趣可以搜索学习
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

def json_print(text):
    # 格式化输出，缩进4个单位
    print(json.dumps(text, sort_keys = True, indent = 4))

def read_data_from_dir(dirname):
    # 文档内容集合
    text = []
    # 遍历文档
    for root, dirs, files in os.walk(dirname):
        for file in files:
            # print(file)
            # 获取文件名
            filename = os.path.splitext(file)[0]
            # 读取文件
            content = json.load(open(root + '/' + file, 'r', encoding = 'utf-8-sig'))
            # 文件名添加到文件中，方便后续生成中间件
            content['FileName'] = filename
            text.append(content)
        
    return text

def keywords_extraction(text):
    r = Rake()
    r.extract_keywords_from_text(text['Text'])
    json_print(text)
    print("==============================")
    print(r.get_ranked_phrases())
    print("==============================")
    print(r.get_ranked_phrases_with_scores())
    print("==============================")
    print(r.stopwords)
    print("==============================")
    print(r.get_word_degrees())

if __name__ == '__main__':
    text = read_data_from_dir('news_1')
    keywords_extraction(text[0])
    #for i in range(5):
        #print(text[i]['Headline'])