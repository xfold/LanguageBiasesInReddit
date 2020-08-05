# RedditBias-ICWSM2021
This repository contains the source code of the original paper ["Discovering and Categorising Language Biases in Reddit"]() accepted at the International Conference on Web and Social Media (ICWSM 2021). This work is part of the project [Discovering and Attesting Digital Discrimination (DADD)](http://dadd-project.org/). 
Related to this work, we created the [Language Bias Visualiser](https://xfold.github.io/WE-GenderBiasVisualisationWeb/), an interactive web-based platform that helps exploring different gender biases found in various datasets.

<i>In this work we present a data-driven approach using word embeddings to discover and categorise language biases on the discussion platform Reddit. As spaces for isolated user communities, platforms such as Reddit are increasingly connected to issues of racism, sexism and other forms of discrimination. Hence, there is a need to monitor the language of these groups. One of the most promising AI approaches to trace linguistic biases in large textual datasets involves word embeddings, which transform text into high-dimensional dense vectors and capture semantic relations between words. Yet, previous studies require predefined sets of potential biases to study, e.g., whether gender is more or less associated with particular types of jobs. This makes these approaches unfit to deal with smaller and community-centric datasets such as those on Reddit, which contain smaller vocabularies and slang, as well as biases that may be particular to that community. This paper proposes a data-driven approach to automatically discover language biases encoded in the vocabulary of online discourse communities on Reddit. In our approach, protected attributes are connected to evaluative words found in the data, which are then categorised through a semantic analysis system. We verify the effectiveness of our method by comparing the biases we discover in the Google News dataset with those found in previous literature. We then successfully discover gender bias, religion bias, and ethnic bias in different Reddit communities. We conclude by discussing potential application scenarios and limitations of this data-driven bias discovery method.</i>


# Overview
<ul>
  <li></li>
  <li></li>
  <li></li>
  <li></li>
  <li></li>
  <li></li>
</ul>

# Setup
First, download or clone the repository. Then, you need to install all dependencies and libraries:
```python
pip3 install -r requirements.txt
```
Ready to run a toy experiment and see if everything is working (Python 3). This command will train a model for a small toy dataset collected from TheRedPill included in the project, estimate its gender biases towards women and men, and cluster them in concepts. Note that this process is only for testing if everything is working as expected, for more specific and accurate biases bigger datasets are required.
```python
python3 Run.py
```
Also, take a look at the python notebook (.ipynb), in which present an step by step exploration of the results.
```python
jupyter notebook
```

# Datasets
We prepared smaller versions of the datasets used in this work consisting of 1M comments. Conceptual and USAS biased in these lightweight versions should be similar to the ones presented in the original paper (which considered the whole dataset). The original datasets were collected using the [PushShift data platform](https://pushshift.io/).

[Dating Advice short](https://osf.io/3rzkb) scrapped from [here](https://www.reddit.com/r/dating_advice/)<br>
[Atheism short](https://osf.io/v2wrg) scrapped from [here](https://www.reddit.com/r/atheism/)<br>
[The Donald short](https://osf.io/g8wsz) scrapped from [here](https://www.reddit.com/r/the_donald/)<br>
[The Red Pill short](https://osf.io/vn6cu) scrapped from [here](https://www.trp.red/feed/) <br>


# Experiments



### Creating your own embeddings model

### Finding biased Words

### Clustering words into concepts
### USAS categories

# Contact
You can find us on our website on [Discovering and Attesting Digital Discrimination](http://dadd-project.org/), or at [@DADD_project](https://twitter.com/DADD_project).
