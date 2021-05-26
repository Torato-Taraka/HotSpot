# HotSpot
This is a framework to extract hot topic from news, it's not a **tool** but a **hub** between the spider and the visualization in the front end.

directory framework:

```
|--news(to put the raw news data in json form)
|  |--xxxx-xx-xx
|  |--xxxx-xx-xx
|  |--xxxx-xx-xx
|  |--...
|--data(auto generated, to store the cleared data and doc vector)
|  |--xxxx-xx-xx.csv
|  |--xxxx-xx-xx.txt
|  |--xxxx-xx-xx.csv
|  |--xxxx-xx-xx.txt
|  |--...
|--knowledge(auto generated, the data to build knowledge graph)
|  |--xxxx-xx-xx-x.txt
|  |--xxxx-xx-xx-x.txt
|  |--...
|--topic(auto generated, the data to visualize)
|  |--xxxx-xx-xx-x.json
|  |--xxxx-xx-xx-x.json
|  |--...
|--environment.yml
|--local.model
|--hotspot.py
|--pre.py
|--process.py(main)
|--others are the test files in the program process
```



# You should

You should create a environment with the `environment.yml` at first

```
conda create -f environment.yml
```

Then, directly use

```
./process.py
```

to generate the intermediate files.

You can not directly see the results unless you click into the directory and  find the lines in the csv file corresponding to the raw news



# How it works

The overall method used in the framework is: use gensim to train local word2vec, and combine the results with GloVe to generate document vectors, then cluster the documents through {DBSCAN, SpectralClustering}, and use sumy to extract abstracts to form titles, and finally generate a heat score.

### 1 Raw data preprocessing

**Input**: news json data
**Output**: data after text processing and information filtering

**step:**
1.1 Limit the range of seven days, load the processed data of the specified date from  `data`, and regenerate it if not

1.2 When regenerating, load the original data of the specified date from news

1.3 Add the file name to the data, convert Chinese symbols to English, lowercase the text, remove redundant items and other text operations

1.4 Extract keywords

1.5 word segmentation

1.6 Clause

1.7 Add keywords, word segmentation, and clauses to the data to form processed data 

### 2 Document vector generation

**Input**: basic processed data
**Output**: document vector

**step:**
2.1 Load the local word2vec model, generate if not, use the word segmentation information in the processed data

2.2 Fusion of local word2vec and GloVe with a certain weight as the word vector of the word

2.3 Find the average of the word vectors of all words in the document as the word vector of the document 

### 3 Clustering

**Input:** document vector
**Output:** clustering result and number of clusters

**step:**
3.1 Use sklearn's DBSCAN or SpectralClustring for unsupervised clustering

3.2 Use Calinski Harabaz score and Davies Bouldin Index (DBI) to evaluate clustering results

3.3 Select the optimal clustering parameters to regenerate the clusters 

### 4 Topic summary

**Input:** clustering result and number of clusters
**Output:** the title of each cluster

**step:**
4.1 Fusion of document titles under the same cluster

4.2 Extract summary with sumy 

### 5 Heat score generation

**Input:** clustering result and number of clusters
**Output:** the popularity index of each cluster

**step:**
5.1 Calculate the current average time of the news of each cluster, in hours, use the formula $\dfrac{1}{1+\beta^{time}}$ to calculate the time score` time_score`, where $\beta$ is the hyperparameter, and `time` is the average hourly distance

5.2 Calculate the number of news in each cluster, take the largest cluster as a reference, use the formula $\dfrac{1+log(x)}{1+log(y)}$ to calculate the number score `number_score`, where `x` is the number of news in a certain cluster, and `y` is the maximum number of cluster news

5.3 Calculate the number of news sources in each cluster, refer to the total number of sources, and use the formula to refer to 5.2 to calculate the source score `source_score`

5.4 Use the formula $\alpha * time\_score + \beta * number\_score + \theta * source\_score$ to calculate the hot score, where $\alpha+\beta+\theta=1, 0<\alpha,\beta,\theta<1 $

### 6 Subsequent data generation

**Input:** data after summary, number of clusters, date, popularity
**Output:** json file containing titles, related articles, popularity index, and txt files containing all sentences in each cluster

**step:**

6.1  Perform the following processing by cluster

6.2 Specify the file name filename as "year-month-day-cluster"

6.3 Collect all sentences, add titles and write them in filename.txt

6.4 Connect the Headline+URL of all articles under the category into a string as detail

6.5 Take out the tuple from the keywords of all articles under the category, merge the same keywords, calculate the weights, and store them in the form of (keywords, key values) as a word cloud

6.6 Write the detail, date, popularity index, word cloud, and title into filename.json in json format 