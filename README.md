# Discovering and Categorising Language Biases in Reddit
This repository contains the source code of the original paper ["Discovering and Categorising Language Biases in Reddit"]() accepted at the International Conference on Web and Social Media (ICWSM 2021). This work is part of the project [Discovering and Attesting Digital Discrimination (DADD)](http://dadd-project.org/). 
Related to this work, we created the [Language Bias Visualiser](https://xfold.github.io/WE-GenderBiasVisualisationWeb/), an interactive web-based platform that helps exploring different gender biases found in various datasets.

<i>In this work we present a data-driven approach using word embeddings to discover and categorise language biases on the discussion platform Reddit. As spaces for isolated user communities, platforms such as Reddit are increasingly connected to issues of racism, sexism and other forms of discrimination. Hence, there is a need to monitor the language of these groups. One of the most promising AI approaches to trace linguistic biases in large textual datasets involves word embeddings, which transform text into high-dimensional dense vectors and capture semantic relations between words. Yet, previous studies require predefined sets of potential biases to study, e.g., whether gender is more or less associated with particular types of jobs. This makes these approaches unfit to deal with smaller and community-centric datasets such as those on Reddit, which contain smaller vocabularies and slang, as well as biases that may be particular to that community. This paper proposes a data-driven approach to automatically discover language biases encoded in the vocabulary of online discourse communities on Reddit. In our approach, protected attributes are connected to evaluative words found in the data, which are then categorised through a semantic analysis system. We verify the effectiveness of our method by comparing the biases we discover in the Google News dataset with those found in previous literature. We then successfully discover gender bias, religion bias, and ethnic bias in different Reddit communities. We conclude by discussing potential application scenarios and limitations of this data-driven bias discovery method.</i>


# Overview
This repository contains the next files and folders:
<ul>
  <li><b>Datasets/</b>: Folder containing the toy dataset <i>toy_1000_trp.csv</i>. Other datasets should be downloaded (see below).</li>
  <li><b>Models/</b>: Folder containing the toy model for dataset <i>toy_1000_trp.csv</i>.</li>
  <li><b>DADDBias_ICWSM.py</b>: Library file which contains all functions to train and test for biases in a corpus.</li>
  <li><b>Run.py</b>: Test execution in python3.</li>
  <li><b>NotebookRun.ipynb</b>: Jupyter notebook with examples on how to train and test the approach.</li>
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
```python
jupyter notebook
```

# Experiments
### Datasets
We prepared smaller versions of the datasets used in this work consisting of 1M comments. Conceptual and USAS biased in these lightweight versions should be similar to the ones presented in the original paper (which considered the whole dataset). The original datasets were collected using the [PushShift data platform](https://pushshift.io/).

[The Red Pill short](https://osf.io/vn6cu) scrapped from [here](https://www.trp.red/feed/) <br>
[Dating Advice short](https://osf.io/3rzkb) scrapped from [here](https://www.reddit.com/r/dating_advice/)<br>
[Atheism short](https://osf.io/v2wrg) scrapped from [here](https://www.reddit.com/r/atheism/)<br>
[The Donald short](https://osf.io/g8wsz) scrapped from [here](https://www.reddit.com/r/the_donald/)<br>



### Creating your own embeddings model

### Finding biased words towards target sets

### Clustering words into concepts
### USAS categories

# Contact
You can find us on our website on [Discovering and Attesting Digital Discrimination](http://dadd-project.org/), or at [@DADD_project](https://twitter.com/DADD_project).
<br>
<i>Updated Aug 2020</i>
