Copyright (c) 2019 [Sebastian Raschka](sebastianraschka.com)

https://github.com/rasbt/python-machine-learning-book-3rd-edition

# Chapter 1 - Giving Computers the Ability to Learn from Data

### Overview

- [Building intelligent machines to transform data into knowledge](#Building-intelligent-machines-to-transform-data-into-knowledge)
- [The three different types of machine learning](#The-three-different-types-of-machine-learning)
    - [Making predictions about the future with supervised learning](#Making-predictions-about-the-future-with-supervised-learning)
        - [Classification for predicting class labels](#Classification-for-predicting-class-labels)
        - [Regression for predicting continuous outcomes](#Regression-for-predicting-continuous-outcomes)
    - [Solving interactive problems with reinforcement learning](#Solving-interactive-problems-with-reinforcement-learning)
    - [Discovering hidden structures with unsupervised learning](#Discovering-hidden-structures-with-unsupervised-learning)
        - [Finding subgroups with clustering](#Finding-subgroups-with-clustering)
        - [Dimensionality reduction for data compression](#Dimensionality-reduction-for-data-compression)
        - [An introduction to the basic terminology and notations](#An-introduction-to-the-basic-terminology-and-notations)
- [A roadmap for building machine learning systems](#A-roadmap-for-building-machine-learning-systems)
    - [Preprocessing - getting data into shape](#Preprocessing--getting-data-into-shape)
    - [Training and selecting a predictive model](#Training-and-selecting-a-predictive-model)
    - [Evaluating models and predicting unseen data instances](#Evaluating-models-and-predicting-unseen-data-instances)
- [Using Python for machine learning](#Using-Python-for-machine-learning)
- [Installing Python packages](#Installing-Python-packages)
- [Summary](#Summary)

## Machine Learning (ML)

- a subfield of Artificial Integlligence (AI)
- application and science of algorithms that make sense of data
- one of the most exciting fields in the CS!
- with abundant of data (~ 2MB data created per second per person!), ML algorithms can be used to spot patterns in data and make predictios about the future events

## Building intelligent machines/algorithms to transform data into knowledge

- data is abundant in both structured and unstructed form
- ML involves self-learning algorithms that derive knowledge from data in order to make predictions
- ML applications are already ubiquitous:
    - spam filter
    - web search engines
    - Network intrusion detection and prevention system
    - digital assistants (Apple Siri, Amazon Alexa, Google Assistant, Microsoft Cortana, etc.)
    - self-driving cars (Google, Uber, Tesla, etc.)
    - skin cancer detection (https://www.nature.com/articles/nature21056)
    - AlphaZero from DeepMind has beaten human champions in Chess, Go and Shogi (Japanese Chess)
  

## Three different types of machine learning

![ML Types](./images/01_01.png)

## Supervised learning

- goal is to learn a model from labeled training data that allows us to make predictions about unseen/unlabeled data

![supervised learning](./images/SupervisedLearning.jpg)

- two types: **classification** and **regression**

### Classification

- classify samples/data to fixed discreted class (labels)
- can be binary class classification
    - e.g. labeling emails as spam or ham, classify internet traffic as malicious or benign, classify programs as malware or benign, classify tumor or benign, etc.
- can bee multi-class classification
    - classify type of malware (adware, spyware, trojans, keyloggers, rootkits, etc.)
    - classify type various stages of cancer (stage 0-4)
    - classify images of people, etc.
- the following figure depicts binary classification problem between **A** and **B** samples    
![Binary Classification](./images/BinaryClassification.jpg)

### Regression

- the outcome is continous value
    - e.g. predicting stock prices, home prices, weather prediction (max/min temps, wind speed, humidity), etc.
- the following figure depicts predicting continuous outcomes given some data
- given a feature variable, $x$, and a targe variable $y$, we fit a straight line to this data that minimizes the distance between the data points and the fitted line

![Continous Outcomes](./images/regression.jpg)

## Reinforcement Learning

- goal is to develop a system (agent) that improves its performance based on interactions with the environment
    - loosely speaking, this is AI
    - similar to how humans and animals with intelligence learn

- the heart of the system is so-called **reward signal**
    - reward can be positive (good) or negative (bad)
    - e.g. training dogs by giving rewards
- the agent interacts with the environment and learn a series of actions by maximizing the rewards via trial-and-error or planning
- the following figure depicts reinforcement learning

![Reinforcement Learning](./images/01_05.png)

## Unsupervised Learning

- deal with unlabelled data or data of unknown structure
- technique can be used to extract meaningful information without the guidance of a known outcome variable or reward function

### Finding subgroups with clustering

- finding natural or meaningful subgroups (**clusters**) without having any prior knowledge
- the following figure illustrates clustering of data into 3 sub-groups

![Clustering](./images/01_06.png)

## Basic terminology and notations

### Dataset and representation

- most ML algorithms learn from data
- classic example of machine learning dataset is the Iris dataset
- contains the measurements of 150 Iris flowers from three species: Setosa, Versicolor and Virginica
- the measuresments are also called the features
- dataset is typically 2-dimensional matrix
- each row represents a sample/observation/instance and each column is a feature value for that sample
- the following figure represents a slice of the Iris dataset


### Irisi Dataset

![Iris Setosa](./images/iris_flowers.jpeg)

![Iris Dataset](./images/01_08.png)


- Iris dataset with 150 samples and four features can be written as a $150x4$ matrix:

    $
    \begin{equation*}
    \textbf{X} \in {\mathbb{R}}^{150x4} : \textbf{X}^{150x4} =
    \begin{bmatrix}
    x_{1}^{1} & x_2^{1} & x_3^{1} & x_4^{1} \\
    x_{1}^{2} & x_2^{2} & x_3^{2} & x_4^{2} \\
    \vdots & \vdots & \vdots & \vdots \\
    x_{1}^{150} & x_2^{150} & x_3^{150} & x_4^{150}
    \end{bmatrix}
    \end{equation*}
    $



### Notational Conventions
- List of Mathematical Symbols: https://oeis.org/wiki/List_of_LaTeX_mathematical_symbols
- superscript $i$ refers to the $i^{th}$ sample/example
- subscript $j$ refers to the $j^{th}$ dimension/feature of the sample
- lowercase, bold-face letters refer to a single vector $\textbf{x} \in \mathbb{R}^{1xm}$
- uppercase, bold-face letters refer to a matrix $\textbf{X} \in \mathbb{R}^{nxm}$
- single element in a vector represented in italics: $\textit{x}_{m}$
- single element in a matrix represented in italics: $\textit{x}_m^{(n)}$

### E.g.
- $x_1^{(150)}$ refers to the first dimension of flower sample #150 **(sepal length)**
- $i^{th}$ flower sample in the matrix can be written as a 4-dimensional row vector:

    $ \textbf{x}^{(i)} \in \mathbb{R}^{1x4}: \textbf{x}^{(i)} = \begin{bmatrix} x_1^{(i)} & x_2^{(i)} & x_3^{(i)} & x_4^{(i)}\end{bmatrix}$
    
- $j^{th}$ feature in matrix is a 150-dimensional column vector:

    $\begin{equation*}
    \textbf{x}_{j} \in \mathbb{R}^{150x1}: \textbf{x}_{j} = 
    \begin{bmatrix} 
       x_j^{(1)} \\
       x_j^{(2)} \\
       \vdots\\
       x_j^{(150)}
    \end{bmatrix}
    \end{equation*}
    $
    
- target variables is 150-dimensional column vector:

$\textbf{y} = \begin{bmatrix} 
    y^{(1)} \\
    y^{(2)} \\
    \vdots \\
    y^{(150)}
    \end{bmatrix}
    (y \in \{Setosa, Versicolor, Virginica\})
$

### Training sample
- a row in a table representating the dataset also called observation, record, instance

### Training
- Model fitting, similar to parameter estimation

### Feature
- a column in a data table (matrix), also called variable, attribute or covariate, input

### Target
- also called outcome, class, label, response variable, ground truth, dependent variable

### Loss function
- also called cost function, error function

## Public Repository of Machine Learning Datasets

1. UCI Machine Learning Repository - https://archive.ics.uci.edu/ml/datasets.php
2. UNB - Canadian Institute for Cybersecurity Datasets - https://www.unb.ca/cic/datasets/index.html
3. Kaggle Public Datasets - https://www.kaggle.com/datasets
4. Sci-kit learn Real-World Datasets -  https://scikit-learn.org/stable/datasets/real_world.html
5. Seaborn Datasets - https://github.com/mwaskom/seaborn-data
6. https://www.openml.org
7. https://huggingface.co/

# A roadmap for building machine learning systems

- the following figure shows a typical workflow for using machine learning in predictive modeling

![Overview of ML systems](./images/MLProcessOverview.jpg)

## Preprocessing - getting data into shape

- raw data can be structured (csv, xml, json, database, etc.) or unstructured (text, images, videos, audios, etc.)
- it's important to convert raw data into feature vector by extracting appropriate features
    - e.g. in Iris dataset, sepal and petal length and width
- feature values are preferred to be on the same scale for optimal performance typically in the range [0, 1]
    - standard normal distribution with zero mean and unit variance (covered in later chapters)
- features may be highly correlated and therefore redundant to a certain degree
    - feature selection and dimensionality reduction techniques are used
    - requires less memory and CPU time
    - may improve predictive performance of model
    - reduce noise (remove irrelevant features)
- dataset is typically divided into two parts (training set and testing/validation set)
- **training set** is used to train the model
- **testing/validation set** is used to evaulate the model to see how it would perform or generalize to new data
- stratified cross-validation techniques are quite common as well when dataset is scarce (small)
    - e.g. 10-fold cross-validation is a common practice

## Training and selecting a predictive model

- each classifier or ML algorithm may have its inherent biases on different type of problem and dataset
- in practice, it's essential to comapre at least a handful of different algorithms in order to train and select the best performing model
- typically certain pre-defined metrics are used to compare performance
    - e.g. accuracy (proportion of correctly classified instances) and many others (covered later...)
- default parameters used by algorithms may work well but not guaranteed
- these hyperparameters may need to be fine-tuned for optimal performance

## Evaluating models and predicting unseen data instances
- the best model, that has been fitted on the training set, is evaluated against the unseen data to estimate generalization error
- if satisfied with the model's performance, it's then deployed and applied to the new, unknown dataset in the real-word
- same scaling techniques (normalization) and dimensionality reduction technique and the parameters are latter reapplied to transform the new data instances

## Why Python for machine learning

- active developers and open source community producing a large number of useful libraries
- NumPy, SciPy on lower-layer uses Fortran and C implementations for fast vector operations on multidimensional arrays
- scikit-learn library has all the popular open-soruce machine learning algorithms
- several deep learning frameworks (TensorFlow, Fast.ai, PyTorch) have Python extensions or built with Python


```python

```
