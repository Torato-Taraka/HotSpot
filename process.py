from hotspot import HotSpot
import pre
import os
import datetime
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def json_pre_process(directory, glove):
    """

    Parameters
    ----------
    directory : str
        the path to json files
    glove : GloVe

    Returns
    -------
    data : pd.DataFrame
        processed data including keywords extracted, word tokenized, sentence tokenized, doc vector

    """
    print('Loading json...')
    # 载入json文件，text为源文件字符处理后的结果，data为text的DataFrame格式
    text, data = pre.load_json_data_from_dir(directory)
    print('extracting keywords...')
    # 提取关键词
    phrases_list = pre.extract_key_phrases_from_doc(text)
    data['Keywords'] = phrases_list
    print('tokenizing...')
    # 分词
    words = pre.words_tokenize(text)
    data['Words'] = words
    # 分句
    data['Sentences'] = pre.sentences_tokenize(text)
    print('vectoring...')
    # 计算文档向量
    doc_vector = pre.get_doc_vector(words, glove, alpha=0.2)
    return data, doc_vector

if __name__ == '__main__':
    print('loading glove...')
    #glove = pre.load_glove('glove.840B.300d.txt')
    print('glove ready...')
    
    news_path = 'news/'
    data_path = 'data/'
    today = datetime.datetime.now()
    data = pd.DataFrame()
    doc_vector = []
    for i in range(7):
        date = today - datetime.timedelta(days=i)
        
        csv = data_path + '{0}-{1}-{2}.csv'.format(date.year, date.month, date.day)
        txt = csv.replace('.csv', '.txt')
        print('=' * 80)
        print(csv)
        print(txt)
        if os.path.exists(csv) and os.path.exists(txt):
            content = pd.read_csv(csv)
            vector = np.loadtxt(txt, dtype=np.float32)
            if data.empty:
                data = content
            else:
                data = pd.concat([data, content], ignore_index=True)
            if type(doc_vector) == np.ndarray:
                doc_vector = np.vstack((doc_vector, vector))
            else:
                doc_vector = vector
            print('data size: ', data.shape)
            print('doc_vector size: ', doc_vector.shape)
            continue
        
        directory = news_path + '{0}-{1}-{2}'.format(date.year, date.month, date.day)
        print(directory)
        if os.path.exists(directory):
            processed_data, vector = json_pre_process(directory, glove)
            processed_data['Csvname'] = csv
            processed_data.to_csv(csv, index=False, encoding='utf-8')
            np.savetxt(txt, vector, fmt="%f")
            print('csv and txt file building...')
            if data.empty:
                data = processed_data
            else:
                data = pd.concat([data, processed_data], ignore_index=True)
            if type(doc_vector) == np.ndarray:
                doc_vector = np.vstack((doc_vector, vector))
            else:
                doc_vector = vector
            print('data size: ', data.shape)
                
    
    print('=' * 80)
    print('preparation finished')
    print('the first of data:', data.iloc[0])
    print('the data info:', data.info())
    print('doc_vector size:', doc_vector.shape)
    print('=' * 80)
    
    hotspot = HotSpot(data, doc_vector)  
    print('Summaries: ')
    for i in range(hotspot.n_clusters):
        print(i, hotspot.hotscore[i], hotspot.summary[i])
    print('=' * 80)
    
    data = hotspot.data
    
    if 'Summary' not in data.columns:
        data['Summary'] = 0
        for i in range(hotspot.n_clusters):
            data.loc[data.Cluster == i, 'Summary'] = hotspot.summary[i]
    else:
        for i in range(hotspot.n_clusters):
            temp = list(data.loc[data.Cluster == i].Summary.values)
            #print(set(temp))
            most_summary = max(set(temp), key=temp.count)
            percentage = data.loc[data.Summary == most_summary].shape[0] / data.shape[0]
            if percentage > 0.5:
                data.loc[data.Cluster == i, 'Summary'] = most_summary
            else:
                data.loc[data.Cluster == i, 'Summary'] = hotspot.summary[i]
    
    print(data.info())
    
    print('=' * 80)
    print('build knowledge graph...')
    pre.build_knowledge_graph(data, hotspot.n_clusters, today)
    print('knowledge graph build success')
    
    print('=' * 80)
    print('build topic...')
    pre.build_topic(data, hotspot.n_clusters, today, hotspot.hotscore)
    print('topic build success')
    
    for i in set(list(data.Csvname)):
        data_to_store = data.loc[data.Csvname == i].drop(columns=['Cluster'])
        csv = data_to_store.iloc[0].Csvname
        data_to_store.to_csv(csv, index=False, encoding='utf-8')
                
    print('=' * 80)
    print('process finished')