# Discovering and Categorising Language Biases in Reddit
This repository contains the source code of the original paper ["Discovering and Categorising Language Biases in Reddit"](https://arxiv.org/abs/2008.02754) accepted at the International Conference on Web and Social Media (ICWSM 2021). This work is part of the project [Discovering and Attesting Digital Discrimination (DADD)](http://dadd-project.org/). 
Related to this work, we created the [Language Bias Visualiser](https://xfold.github.io/WE-GenderBiasVisualisationWeb/), an interactive web-based platform that helps exploring gender biases found in various Reddit datasets.

<i>In this work we present a data-driven approach using word embeddings to discover and categorise language biases on the discussion platform Reddit. As spaces for isolated user communities, platforms such as Reddit are increasingly connected to issues of racism, sexism and other forms of discrimination. Hence, there is a need to monitor the language of these groups. One of the most promising AI approaches to trace linguistic biases in large textual datasets involves word embeddings, which transform text into high-dimensional dense vectors and capture semantic relations between words. Yet, previous studies require predefined sets of potential biases to study, e.g., whether gender is more or less associated with particular types of jobs. This makes these approaches unfit to deal with smaller and community-centric datasets such as those on Reddit, which contain smaller vocabularies and slang, as well as biases that may be particular to that community. This paper proposes a data-driven approach to automatically discover language biases encoded in the vocabulary of online discourse communities on Reddit. In our approach, protected attributes are connected to evaluative words found in the data, which are then categorised through a semantic analysis system. We verify the effectiveness of our method by comparing the biases we discover in the Google News dataset with those found in previous literature. We then successfully discover gender bias, religion bias, and ethnic bias in different Reddit communities. We conclude by discussing potential application scenarios and limitations of this data-driven bias discovery method.</i>


# Overview
This repository contains the next files and folders:
<ul>
  <li><b>Datasets/</b>: Folder containing the toy dataset <i>toy_1000_trp.csv</i>. Other datasets should be downloaded (see below).</li>
  <li><b>Models/</b>: Folder containing the toy model for dataset <i>toy_1000_trp.csv</i>.</li>
  <li><b>DADDBias_ICWSM.py</b>: Library file which contains all functions to train and test for biases in a corpus.</li>
  <li><b>Run.py</b>: Test execution in python3.</li>
  <li><b>RunNotebook.ipynb</b>: Jupyter notebook with examples on how to train and test the approach.</li>
  <li><b>README.md</b>: This file.</li>
  <li><b>requirements.txt</b>: Requirements file.</li>
</ul>

# Setup
First, download or clone the repository. Then, you need to install all dependencies and libraries:
```python
pip3 install -r requirements.txt
```
Now we are ready to run a toy experiment and see if everything is working (Python 3):
```python
python3 Run.py
```
This command will train a model for a small toy dataset collected from TheRedPill included in the project, estimate its gender biases towards women and men, and cluster them in concepts. <b>Note that this process is only for testing if everything is working as expected, and the results might not be representative of the community explored. In Run.py we are using a set of parameters and a small dataset only to speed-up the testing process.</b>

Also, take a look at the python notebook (.ipynb) included in the project in which present an step by step explanation of the different processes.

# Experiments
### Datasets
We prepared smaller versions of the datasets used in this work consisting of 1M comments. Conceptual and USAS biases in these lightweight versions should be similar to the ones presented in the original paper considering the whole dataset. The original datasets were collected using the [PushShift data platform](https://pushshift.io/).

<ul>
  <li>(The Red Pill short):(https://osf.io/vn6cu) scrapped from [here](https://www.trp.red/feed/) <br></li>
  <li>[Dating Advice short](https://osf.io/3rzkb) scrapped from [here](https://www.reddit.com/r/dating_advice/)<br></li>
  <li>[Atheism short](https://osf.io/v2wrg) scrapped from [here](https://www.reddit.com/r/atheism/)<br></li>
  <li>[The Donald short](https://osf.io/g8wsz) scrapped from [here](https://www.reddit.com/r/the_donald/)<br></li>
</ul>


### Creating your own embeddings model
To create your own embeddings model in order to attest biases you only need to do:
```python
import DADDBias_ICWSM
DADDBias_ICWSM.TrainModel(setup['csvfile'],   #csv file with reddit (or other platform's) comments
  'body',                            # column with the comments
  outputname = setup['outputFile'],  # output model filename
  window = 4,                        # window
  minf = 10,                         # minimum frequency (words less frequent than threshold will be ignored)
  epochs = 100,                      # training epochs
  ndim = 200,                        # embedding dimensions
  verbose = False                    # verbose
  )        
```

After a while, depending on the size of the dataset and parameters used, this command will create the skip-gram model for the dataset.
Note that this code is prepared to load comments from a .csv file, but you can also load comments from any other format by modifying `DADDBias_ICWSM.py`. 

Finally, the larger the dataset, the more stable the resulting embeddings and model will be. For instance, the datasets explored in the paper contain over 2M comments and have an average word density (average unique new words per comment) of around 0.0003.

### Finding biased words towards target sets
Once we have an embedding model trained and two target sets (lists of words) we want to analyse, we call:
```python
import DADDBias_ICWSM
[b1,b2] = DADDBias_ICWSM.GetTopMostBiasedWords(
  modelpath,                         # path to the embedding model
  300,                               # top k biased words to discover
  ts1,                               # target set 1
  ts2,                               # target set 2
  ['JJ','JJR','JJS'],                # interesting parts of speech to analyse
  True)                              # verbose
```
This function returns two word lists of words, `b1` and `b2`, which contain all words from the embedding model most biased towards target set 1 (`ts1`) and target set 2 (`ts2`), respectively. Here, the target sets should be any lists of words that help describe a concept. For instance, in our work we utilise lists of words used in previous research, such as:
```python
women = ["sister" , "female" , "woman" , "girl" , "daughter" , "she" , "hers" , "her"]
men   = ["brother" , "male" , "man" , "boy" , "son" , "he" , "his" , "him"]  
```

Finally, both `b1` and `b2` are lists of word objects such that `b1 = [w1, w2, ..., w300]`, and every word contains next attributes:
```python
w1 = {
  'word' : 'casual'                  # Word 
  'bias' : 0.217                     # Bias strength towards target set 1 (in this example) when compared to target set 2
  'freq' : 6773                      # Frequency of word in the vocabulary of the model
  'pos'  : 'JJ'                      # Part of speech as determined by NLTK
  'wv'   : np.array(..)              # Embedding of the word, used for clustering later
  'rank' : 1834                      # Frequency ranking of the word in model's vocabulary
  'sent' : 0.2023                    # Sentiment of word [-1,1], as determined by nltk.sentiment.vader
}
```
By exploring the different properties and values of the most biased words towards a target set when compared to another target set in that community, we can get some interesting insights about the community itself and how the two target sets compare!

### Clustering words into concepts
In order to aggregate words into concepts and to discover the <i>conceptual biases</i> of the community towards both target sets, we use:
```python
import DADDBias_ICWSM
[cl1,cl2] = DADDBias_ICWSM.Cluster(
  b1,                                # set of baised words towards target set 1 
  b2,                                # set of baised words towards target set 2
  0.15,                              # r value to decide k-means k; k = r* avg(len(b1), len(b2))
  100                                # number of times to repeat the k-means clustering, only keeping the partition with best intrasim value
)
```
This process leverages the embeddings of the different set of biased words to aggregate words that are close in the embedding space. The return values of the function are the partition with best intrasim found biased towards target set 1 `cl1`, and for target set 2 `cl2`. Some cluster examples from r/TheRedPill:
```python
#biased towards target set women
for cluster in cl1:
    print( [k['word'] for k in cluster] )
    
>> ['promiscuous', 'promiscous']
>> ['lesbian', 'bisexual', 'polyamorous']
>> ['erratic', 'solipsistic', 'unreasonable', 'illogical', 'arbitrary', 'unrealistic']
>> ['hot', 'gorgeous']
>> ['exclusive', 'monogamous']
>> ...

#biased towards target set men
for cluster in cl2:
    print( [k['word'] for k in cluster] )
    
>> ['unapologetic', 'authentic']
>> ['genious', 'pappy', 'disciplinarian', 'lowliest', 'venusian', 'venerable', ...]
>> ['visionary', 'tactician', 'eccentric', 'courageous', 'charasmatic', ...]
>> ['influential', 'powerful']
>> ['charismatic', 'authoritative', 'assertive']
>> ...
```

### USAS categories
In the paper, after identifying the conceptual biases of the community towards the different target sets, we label each one of these clusters by using the UCREL Semantic Analysis System, also named USAS. USAS is <i>a framework for the automatic semantic analysis and tagging of text, originally based on Tom McArthur’s Longman Lexicon of Contemporary English (more information on the paper)</i>. The USAS tagger forms part of Wmatrix, a paid tool for corpus analysis, however the authors also offer a free online version [here](http://ucrel-api.lancaster.ac.uk/usas/tagger.html). 

In our work, we automated the crawling of the most frequent label in each cluster and partition with a simple script while taking care not to affect the performance of the online free version. However, in order to discourage the automation of this practice we are not sharing this part of the code; the same results can be obtained using the online version of the USAS tagger normally.

# Contact
You can find us on our website on [Discovering and Attesting Digital Discrimination](http://dadd-project.org/), or at [@DADD_project](https://twitter.com/DADD_project).
<br>
<i>Updated Aug 2020</i>
