- Note - Don't run the cells as a live demo - some tasks can take 10 minutes or longer...

# Text Classification

- applying machine learning to classify natural language for various tasks
- a comprehensive article on Text Classification: https://arxiv.org/pdf/2004.03705.pdf
- some common text classification tasks:
    1. sentiment analysis
    2. news categorization
    3. topic analysis
    4. question answering (QA)
    5. natural language inference (NLI)
    
## Sentiment Analysis
- subfield of **natural language processing (NLP)** 
- also called **opinion mining**
- apply ML algorithms to classify documents based on their polarity:
    - the attitude of the writer

## General steps
1. clean and prepare text data
2. build feature vectors from text documents
3. train a machine learning model to classify positive and negative movie reviews
4. test and evaluate the model

## IMDb dataset
- contains 50,000 labeled movie reviews from the Internet Movie Database (IMDb)
- task is to classify reviews as **positive** or **negative**
- compressed archive can be downloaded from: http://ai.stanford.edu/~amaas/data/sentiment


### Download and untar IMDb dataset
- on Linux and Mac use the following cells
- on Windows, manually download the archive and untar using 7Zip or other applications
- or use the provided Python code


```bash
%%bash
# let's download the file
# FYI - file is ~ 84 MB; may take a while depending on Internet speed...
# Extracting files from a Tar file may take even longer...
dirPath=data
fileName=aclImdb_v1.tar.gz
url=http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
if [ -f "$dirPath/$fileName" ]; then
    echo "File $dirPath/$fileName exists."
else
    echo "File $dirPath/$fileName does not exist. Downloading from $url..."
    mkdir -p "$dirPath"
    curl -o "$dirPath/$fileName" "$url"
    cd $dirPath
    tar -xf "$fileName"
fi
```


```python
# let's see the contents of the data folder
! ls data
```


```python
# let's untar the compressed aclImdb_v1.tar.gz file
! tar -zxf data/aclImdb_v1.tar.gz --directory data
```

### Python code to download and extract tar file
- this can take a while depending on the Internet speed...


```python
import os
import sys
import tarfile
import time
import urllib.request


source = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
target = 'data/aclImdb_v1.tar.gz'

def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = progress_size / (1024**2 * duration)
    percent = count * block_size * 100 / total_size

    sys.stdout.write("\r%d%% | %d MB | %.2f MB/s | %d sec elapsed" %
                    (percent, progress_size / (1024**2), speed, duration))
    sys.stdout.flush()


if not os.path.isdir('data/aclImdb') and not os.path.isfile(target):
    urllib.request.urlretrieve(source, target, reporthook)
```


```python
# untar the file
if not os.path.isdir('data/aclImdb'): # if the directory doesn't exist untar the target to path
    with tarfile.open(target, 'r:gz') as tar:
        tar.extractall(path="./data")
else:
    print('data/aclImdb folder exists!' )
```

### Preprocess the movie dataset into a more convenient format

- extract and load the movie dataset into Pandas DataFrame
- NOTE: can take up to **10 minutes** on a PC
- use the Pthon Progress Indicator (PyPrind) package to show the progress bar from Python code


```python
! pip install pyprind
```


```python
import pyprind
import pandas as pd
import os

# change the `basepath` to the directory of the
# unzipped movie dataset

basepath = 'data/aclImdb'

labels = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(50000)
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        
        path = os.path.join(basepath, s, l)
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path, file), 
                      'r', encoding='utf-8') as infile:
                txt = infile.read()
            df1 = pd.DataFrame([[txt, labels[l]]])
            df = pd.concat([df, df1], ignore_index=True)
            #df = df.append([[txt, labels[l]]], 
            #               ignore_index=True)
            pbar.update()
df.columns = ['review', 'sentiment']
```


```python
df
```

### Shuffle and save the assembled data as CSV file

- pickle the DataFrame as a binary file for faster load


```python
import pandas as pd
import numpy as np
import pickle
```


```python
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index)) # randomize files
```


```python
df
```


```python
# save csv format
df.to_csv('data/movie_data.csv', index=False, encoding='utf-8')
```


```python
# save DataFrame as a pickle dump
pickle.dump(df, open('data/movie_data.pd', 'wb'))
```

## Start Here After the pickle dump of movie_data.pd

- after the first run of the above cells load the pickle file directly from the cell below


```python
# directly load the pickled file as DataFrame
import pandas as pd
import numpy as np
import pickle
df = pickle.load(open('data/movie_data.pd', 'rb'))
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11841</th>
      <td>In 1974, the teenager Martha Moxley (Maggie Gr...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19602</th>
      <td>OK... so... I really like Kris Kristofferson a...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>45519</th>
      <td>***SPOILER*** Do not read this, if you think a...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25747</th>
      <td>hi for all the people who have seen this wonde...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>42642</th>
      <td>I recently bought the DVD, forgetting just how...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21243</th>
      <td>OK, lets start with the best. the building. al...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>45891</th>
      <td>The British 'heritage film' industry is out of...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>42613</th>
      <td>I don't even know where to begin on this one. ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>43567</th>
      <td>Richard Tyler is a little boy who is scared of...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2732</th>
      <td>I waited long to watch this movie. Also becaus...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>50000 rows × 2 columns</p>
</div>



### bag-of-words model

- ML algorithms only work on numerical values
- need to encode/transform text data into numerical values using **bag-of-words** model
- **bag-of-words** technique allows us to represent text as numerical feature vectors:
    1. extract all the unique tokens -- e.g., words -- from the entire document
    2. construct a feature vector that contains the word frequency in the particular document 
    3. order of the words in the document doesn't matter - hence bag-of-words
- since the unique words in each document represent only a small subset of all the words in the bag-of-words vocabulary, the feature vector will be **sparse** mostly consisting of zeros

### transform words into feature vectors

- use `CountVectorizer` class implemented in scikit-learn
- https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
- `CountVectorizer` takes an array of text data and returns a bag-of-words vectors


```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer()
docs = np.array([
        'The sun is shining',
        'The weather is sweet',
        'The sun is shining, the weather is sweet, and one and one is two'])
bag = count.fit_transform(docs)
```


```python
# let's look at the vocabulary_ contents of count object
count.vocabulary_
```




    {'the': 6,
     'sun': 4,
     'is': 1,
     'shining': 3,
     'weather': 8,
     'sweet': 5,
     'and': 0,
     'one': 2,
     'two': 7}




```python
bag
```




    <3x9 sparse matrix of type '<class 'numpy.int64'>'
    	with 17 stored elements in Compressed Sparse Row format>




```python
bag.toarray()
```




    array([[0, 1, 0, 1, 1, 0, 1, 0, 0],
           [0, 1, 0, 0, 0, 1, 1, 0, 1],
           [2, 3, 2, 1, 1, 1, 2, 1, 1]])



### bag-of-words feature vector

- the values in the feature vectors are also called the **raw term frequencies**
    - $x^i = tf(t^i, d)$
    - the number of times a term, $t$ appears in a document, $d$
- indices of terms are usually assigned alphabetically

### N-gram models

- the above model is **1-gram** or **unigram** model
    - each item or token in the vocabulary represents a single word
- if the sentence is: "The sun is shining"
    - **1-gram**: "the", "sun", "is", "shining"
    - **2-gram**: "the sun", "sun is", "is shining"
- `CountVectorizer` class allows us to use different n-gram models via its `ngram_range` parameter
- e.g. ngram_range(2, 2) will use 2-gram model

## Assess word relevency via term frequency - inverse document frequency

- words often occur across multiple documents from all the classes (positive and negative in IMDb)
- frequently occuring words across classes don't contain discriminatory information
- **tf-idf** model can be used to down weight these frequently occurring words in the feature vectors
    
    $$\text{tf}\mbox{-}\text{idf}(t, d) = \text{tf}(t, d) \times \text{idf}(t, d)$$
    $$\text{idf}(t, d) = log\frac{n_d}{1+\text{df}(d, t)}$$
    - $n_d$ - total number of documents
    - $\text{df}(d, t)$ - number of documents, $d$ that contain the term $t$
    - $log$ ensures that low document frequencies are not given too much weight
- scikit-learn implements `TfidfTransformer` class which takes the raw term frequencies from the `CountVectorizer` class as input and returns tf-idf feature vectors


```python
from sklearn.feature_extraction.text import TfidfTransformer
```


```python
tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
np.set_printoptions(precision=2)
tfidf.fit_transform(bag).toarray()
```




    array([[0.  , 0.43, 0.  , 0.56, 0.56, 0.  , 0.43, 0.  , 0.  ],
           [0.  , 0.43, 0.  , 0.  , 0.  , 0.56, 0.43, 0.  , 0.56],
           [0.5 , 0.45, 0.5 , 0.19, 0.19, 0.19, 0.3 , 0.25, 0.19]])



### Note

- `is` (index = 1) has the largest **TF** of $3$ in the third document
- after transforming, **is** now has relatively small tf-idf ($0.45$) in the $3^{rd}$ document
- `TfidfTransformer` calculates `idf` and `tf-idf` slight differently (adds 1)

$$\text{idf} (t,d) = log\frac{1 + n_d}{1 + \text{df}(d, t)}$$
$$\text{tf-idf}(t,d) = \text{tf}(t,d) \times (\text{idf}(t,d)+1)$$

- by default, `TfidfTransformer` applies the L2-normalization (`norm='l2'`), which returns a vector of length 1 by dividing an un-normalized feature vector *v* by its L2-norm

$$v_{\text{norm}} = \frac{v}{||v||_2} = \frac{v}{\sqrt{v_{1}^{2} + v_{2}^{2} + \dots + v_{n}^{2}}} = \frac{v}{\big (\sum_{i=1}^{n} v_{i}^{2}\big)^\frac{1}{2}}$$

- let's see an example of how `tf-idf` is calculated


```python
# unnormalized tf-idf of 'is' in document 3 can be calculated as follows
tf_is = 3
n_docs = 3
idf_is = np.log((n_docs+1) / (3+1))
tfidf_is = tf_is * (idf_is + 1)
print('tf-idf of term "is" = %.2f' % tfidf_is)
```

    tf-idf of term "is" = 3.00


- repeat the calculations for every term in $3^{rd}$ document
    - we'll get a tf-idf vector: [3.39, 3.0, 3.39, 1.29, 1.29, 1.29, 2.0 , 1.69, 1.29]
- let's apply L2-normalization:
$$\text{tf-idf}_{norm} = \frac{[3.39, 3.0, 3.39, 1.29, 1.29, 1.29, 2.0 , 1.69, 1.29]}{\sqrt{[3.39^2+3.0^2+3.39^2+ 1.29^2 + 1.29^2 + 1.29^2 + 2.0^2 + 1.69^2 + 1.29^2]}}$$

$$=[0.5, 0.45, 0.5, 0.19, 0.19, 0.19, 0.3, 0.25, 0.19]$$

$$\Rightarrow \text{tfi-df}_{norm}("is", d3) = 0.45$$


```python
# Calculate tf-idf without normalization
tfidf = TfidfTransformer(use_idf=True, norm=None, smooth_idf=True)
raw_tfidf = tfidf.fit_transform(count.fit_transform(docs)).toarray()[-1]
raw_tfidf 
```




    array([3.39, 3.  , 3.39, 1.29, 1.29, 1.29, 2.  , 1.69, 1.29])




```python
# Now apply l2-normalization
l2_tfidf = raw_tfidf / np.sqrt(np.sum(raw_tfidf**2))
l2_tfidf
# same result as TfidfTransformer with L2-regularization
```




    array([0.5 , 0.45, 0.5 , 0.19, 0.19, 0.19, 0.3 , 0.25, 0.19])



## Cleaning text data

- text may have unwanted characters such as HTML/XML tags and punctuations
- convert all text into lowercase
    - we may lose characteristics of proper nouns, but they're not relevant in sentiment analysis
- remove all unwanted characters but keep emoticons such as: `:) :(` (smiley face, sad face, etc.)
    - emoticons have sentiment values
    - however, remove *nose* character ( `-` in `:-)` ) from the emoticons for consistency
- for simplicity, we use regular expressions; however
    - sophisticated libraries such as BeautifulSoup and Python HTML.parser exist for parsing HTML/XML documents
    - regular expressions are sufficient for this application to clean the unwanted characters
- let's display the last 50 characters from the first document in the reshuffled movie review dataset


```python
df.loc[1, 'review']
```




    'Actor turned director Bill Paxton follows up his promising debut, the Gothic-horror "Frailty", with this family friendly sports drama about the 1913 U.S. Open where a young American caddy rises from his humble background to play against his Bristish idol in what was dubbed as "The Greatest Game Ever Played." I\'m no fan of golf, and these scrappy underdog sports flicks are a dime a dozen (most recently done to grand effect with "Miracle" and "Cinderella Man"), but some how this film was enthralling all the same.<br /><br />The film starts with some creative opening credits (imagine a Disneyfied version of the animated opening credits of HBO\'s "Carnivale" and "Rome"), but lumbers along slowly for its first by-the-numbers hour. Once the action moves to the U.S. Open things pick up very well. Paxton does a nice job and shows a knack for effective directorial flourishes (I loved the rain-soaked montage of the action on day two of the open) that propel the plot further or add some unexpected psychological depth to the proceedings. There\'s some compelling character development when the British Harry Vardon is haunted by images of the aristocrats in black suits and top hats who destroyed his family cottage as a child to make way for a golf course. He also does a good job of visually depicting what goes on in the players\' heads under pressure. Golf, a painfully boring sport, is brought vividly alive here. Credit should also be given the set designers and costume department for creating an engaging period-piece atmosphere of London and Boston at the beginning of the twentieth century.<br /><br />You know how this is going to end not only because it\'s based on a true story but also because films in this genre follow the same template over and over, but Paxton puts on a better than average show and perhaps indicates more talent behind the camera than he ever had in front of it. Despite the formulaic nature, this is a nice and easy film to root for that deserves to find an audience.'




```python
import re
# create a function to do the preprocessing
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text) # remove HTML
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', 
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', '')) # convert upper to lowercase; remove - from :-)
    return text
```


```python
# let's preprocess the above text
preprocessor(df.loc[1, 'review'])
```




    'actor turned director bill paxton follows up his promising debut the gothic horror frailty with this family friendly sports drama about the 1913 u s open where a young american caddy rises from his humble background to play against his bristish idol in what was dubbed as the greatest game ever played i m no fan of golf and these scrappy underdog sports flicks are a dime a dozen most recently done to grand effect with miracle and cinderella man but some how this film was enthralling all the same the film starts with some creative opening credits imagine a disneyfied version of the animated opening credits of hbo s carnivale and rome but lumbers along slowly for its first by the numbers hour once the action moves to the u s open things pick up very well paxton does a nice job and shows a knack for effective directorial flourishes i loved the rain soaked montage of the action on day two of the open that propel the plot further or add some unexpected psychological depth to the proceedings there s some compelling character development when the british harry vardon is haunted by images of the aristocrats in black suits and top hats who destroyed his family cottage as a child to make way for a golf course he also does a good job of visually depicting what goes on in the players heads under pressure golf a painfully boring sport is brought vividly alive here credit should also be given the set designers and costume department for creating an engaging period piece atmosphere of london and boston at the beginning of the twentieth century you know how this is going to end not only because it s based on a true story but also because films in this genre follow the same template over and over but paxton puts on a better than average show and perhaps indicates more talent behind the camera than he ever had in front of it despite the formulaic nature this is a nice and easy film to root for that deserves to find an audience '




```python
# quick test for emoticons
preprocessor("</a>This :) is :( a test :-)! more test :-( <img />")
```




    'this is a test more test :) :( :) :('




```python
# let's preprocess the review column in DataFrame
df['review'] = df['review'].apply(preprocessor)
```


```python
# quick test
df.loc[1000, 'review'][-50:]
```




    ' disappointing to what was actually a great story '



## Processing documents into tokens

- An easy way to *tokenize* documents is to split them into individual words by splitting the cleaned documents using whitespace characters


```python
def tokenizer(text):
    return text.split()
```


```python
tokenizer(' runners like running and thus they run ')
```




    ['runners', 'like', 'running', 'and', 'thus', 'they', 'run']



### Word stemming

- transforming a word into its root form
- allows to map related words typically with the same meaning to the same stem
- **Porter stemmer** is one of the oldest and simplest algorithms used to find the words' stem
- **Porter stemmer** is implemented in the **Natural Language Toolkit (NLTK)**
    - http://www.nltk.org/
- other algorithms found in NLTK are: 
    - **Snowball stemmer (Porter2 or English stemmer)**
    - **Lancaster stemmer**
- must install `nltk` framework to use


```python
! pip install nltk
```

    Requirement already satisfied: nltk in /opt/anaconda3/envs/ml/lib/python3.10/site-packages (3.8.1)
    Requirement already satisfied: click in /opt/anaconda3/envs/ml/lib/python3.10/site-packages (from nltk) (8.1.3)
    Requirement already satisfied: regex>=2021.8.3 in /opt/anaconda3/envs/ml/lib/python3.10/site-packages (from nltk) (2023.3.23)
    Requirement already satisfied: tqdm in /opt/anaconda3/envs/ml/lib/python3.10/site-packages (from nltk) (4.64.1)
    Requirement already satisfied: joblib in /opt/anaconda3/envs/ml/lib/python3.10/site-packages (from nltk) (1.1.1)



```python
from nltk.stem.porter import PorterStemmer
```


```python
porter = PorterStemmer()
```


```python
def porter_stemmer(text):
    # use tokenizer function defined above
    return [porter.stem(word) for word in tokenizer(text)]
```


```python
porter_stemmer('runners like running and thus they run')
```




    ['runner', 'like', 'run', 'and', 'thu', 'they', 'run']



### Stop-words removal

- words that are extremely common in all sorts of texts and probably bear no (or only a little) useful information
- can't help in distinguishing between different classes of documents
    - e.g.: *is, has, and, like, are, am, etc.*
- removing stopwords can reduce the feature vector size without losing important information
- NLTK library has a set of 127 stop-words which can be downloaded using `nltk.download` function


```python
import nltk
```


```python
nltk.download('stopwords')
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /Users/rbasnet/nltk_data...
    [nltk_data]   Unzipping corpora/stopwords.zip.





    True




```python
from nltk.corpus import stopwords
stop = stopwords.words('english')
```


```python
stop
```




    ['a',
     'about',
     'above',
     'after',
     'again',
     'against',
     'ain',
     'all',
     'am',
     'an',
     'and',
     'any',
     'are',
     'aren',
     "aren't",
     'as',
     'at',
     'be',
     'because',
     'been',
     'before',
     'being',
     'below',
     'between',
     'both',
     'but',
     'by',
     'can',
     'couldn',
     "couldn't",
     'd',
     'did',
     'didn',
     "didn't",
     'do',
     'does',
     'doesn',
     "doesn't",
     'doing',
     'don',
     "don't",
     'down',
     'during',
     'each',
     'few',
     'for',
     'from',
     'further',
     'had',
     'hadn',
     "hadn't",
     'has',
     'hasn',
     "hasn't",
     'have',
     'haven',
     "haven't",
     'having',
     'he',
     "he'd",
     "he'll",
     'her',
     'here',
     'hers',
     'herself',
     "he's",
     'him',
     'himself',
     'his',
     'how',
     'i',
     "i'd",
     'if',
     "i'll",
     "i'm",
     'in',
     'into',
     'is',
     'isn',
     "isn't",
     'it',
     "it'd",
     "it'll",
     "it's",
     'its',
     'itself',
     "i've",
     'just',
     'll',
     'm',
     'ma',
     'me',
     'mightn',
     "mightn't",
     'more',
     'most',
     'mustn',
     "mustn't",
     'my',
     'myself',
     'needn',
     "needn't",
     'no',
     'nor',
     'not',
     'now',
     'o',
     'of',
     'off',
     'on',
     'once',
     'only',
     'or',
     'other',
     'our',
     'ours',
     'ourselves',
     'out',
     'over',
     'own',
     're',
     's',
     'same',
     'shan',
     "shan't",
     'she',
     "she'd",
     "she'll",
     "she's",
     'should',
     'shouldn',
     "shouldn't",
     "should've",
     'so',
     'some',
     'such',
     't',
     'than',
     'that',
     "that'll",
     'the',
     'their',
     'theirs',
     'them',
     'themselves',
     'then',
     'there',
     'these',
     'they',
     "they'd",
     "they'll",
     "they're",
     "they've",
     'this',
     'those',
     'through',
     'to',
     'too',
     'under',
     'until',
     'up',
     've',
     'very',
     'was',
     'wasn',
     "wasn't",
     'we',
     "we'd",
     "we'll",
     "we're",
     'were',
     'weren',
     "weren't",
     "we've",
     'what',
     'when',
     'where',
     'which',
     'while',
     'who',
     'whom',
     'why',
     'will',
     'with',
     'won',
     "won't",
     'wouldn',
     "wouldn't",
     'y',
     'you',
     "you'd",
     "you'll",
     'your',
     "you're",
     'yours',
     'yourself',
     'yourselves',
     "you've"]




```python
sentense = 'a runner likes running a lot'
[w for w in porter_stemmer(sentense) if w not in stop]
```




    ['runner', 'like', 'run', 'lot']



## Training a logistic regression model for document classification

- our DataFrame is already randomized; let's just split 
- use the `Pipeline` class implemented in scikit-learn - [https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- Pipeline lets us sequentially apply a list of transforms and a final estimator
- intermediate steps of the pipeline must be `transforms`, 
    - that is, they must implement fit and transform methods
- the final estimator only needs to implement fit
- we'll also use `GridSearchCV` object to find the optimal set of parameters for our logistic regression model


```python
# improve our tokenizer function
def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized
```


```python
# split dataset into 50/50 (just following text)
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values
```


```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

stop = stopwords.words('english')

param_grid = [{'vect__ngram_range': [(1, 1)],
               #'vect__stop_words': [stop, None], # doesn't add to performance
               'vect__tokenizer': [tokenizer],
               'clf__penalty': ['l1', 'l2'],
               'vect__use_idf':[True],
               'vect__norm':[None],
               'clf__C': [1.0, 10.0]},
              {'vect__ngram_range': [(1, 1)],
               #'vect__stop_words': [None] 
               'vect__tokenizer': [tokenizer],
               'vect__use_idf':[False],
               'vect__norm':['l2'],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0]},
              ]

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0, solver='liblinear'))])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=2,
                           n_jobs=-1)
```


```python
gs_lr_tfidf.fit(X_train, y_train)
```

    Fitting 5 folds for each of 12 candidates, totalling 60 fits





<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=5,
             estimator=Pipeline(steps=[(&#x27;vect&#x27;,
                                        TfidfVectorizer(lowercase=False)),
                                       (&#x27;clf&#x27;,
                                        LogisticRegression(random_state=0,
                                                           solver=&#x27;liblinear&#x27;))]),
             n_jobs=-1,
             param_grid=[{&#x27;clf__C&#x27;: [1.0, 10.0], &#x27;clf__penalty&#x27;: [&#x27;l1&#x27;, &#x27;l2&#x27;],
                          &#x27;vect__ngram_range&#x27;: [(1, 1)], &#x27;vect__norm&#x27;: [None],
                          &#x27;vect__tokenizer&#x27;: [&lt;function tokenizer at 0x7fe3cf04ec20&gt;],
                          &#x27;vect__use_idf&#x27;: [True]},
                         {&#x27;...
                          &#x27;vect__stop_words&#x27;: [[&#x27;a&#x27;, &#x27;about&#x27;, &#x27;above&#x27;, &#x27;after&#x27;,
                                                &#x27;again&#x27;, &#x27;against&#x27;, &#x27;ain&#x27;,
                                                &#x27;all&#x27;, &#x27;am&#x27;, &#x27;an&#x27;, &#x27;and&#x27;, &#x27;any&#x27;,
                                                &#x27;are&#x27;, &#x27;aren&#x27;, &quot;aren&#x27;t&quot;, &#x27;as&#x27;,
                                                &#x27;at&#x27;, &#x27;be&#x27;, &#x27;because&#x27;, &#x27;been&#x27;,
                                                &#x27;before&#x27;, &#x27;being&#x27;, &#x27;below&#x27;,
                                                &#x27;between&#x27;, &#x27;both&#x27;, &#x27;but&#x27;, &#x27;by&#x27;,
                                                &#x27;can&#x27;, &#x27;couldn&#x27;, &quot;couldn&#x27;t&quot;, ...],
                                               None],
                          &#x27;vect__tokenizer&#x27;: [&lt;function tokenizer at 0x7fe3cf04ec20&gt;],
                          &#x27;vect__use_idf&#x27;: [False]}],
             scoring=&#x27;accuracy&#x27;, verbose=2)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=5,
             estimator=Pipeline(steps=[(&#x27;vect&#x27;,
                                        TfidfVectorizer(lowercase=False)),
                                       (&#x27;clf&#x27;,
                                        LogisticRegression(random_state=0,
                                                           solver=&#x27;liblinear&#x27;))]),
             n_jobs=-1,
             param_grid=[{&#x27;clf__C&#x27;: [1.0, 10.0], &#x27;clf__penalty&#x27;: [&#x27;l1&#x27;, &#x27;l2&#x27;],
                          &#x27;vect__ngram_range&#x27;: [(1, 1)], &#x27;vect__norm&#x27;: [None],
                          &#x27;vect__tokenizer&#x27;: [&lt;function tokenizer at 0x7fe3cf04ec20&gt;],
                          &#x27;vect__use_idf&#x27;: [True]},
                         {&#x27;...
                          &#x27;vect__stop_words&#x27;: [[&#x27;a&#x27;, &#x27;about&#x27;, &#x27;above&#x27;, &#x27;after&#x27;,
                                                &#x27;again&#x27;, &#x27;against&#x27;, &#x27;ain&#x27;,
                                                &#x27;all&#x27;, &#x27;am&#x27;, &#x27;an&#x27;, &#x27;and&#x27;, &#x27;any&#x27;,
                                                &#x27;are&#x27;, &#x27;aren&#x27;, &quot;aren&#x27;t&quot;, &#x27;as&#x27;,
                                                &#x27;at&#x27;, &#x27;be&#x27;, &#x27;because&#x27;, &#x27;been&#x27;,
                                                &#x27;before&#x27;, &#x27;being&#x27;, &#x27;below&#x27;,
                                                &#x27;between&#x27;, &#x27;both&#x27;, &#x27;but&#x27;, &#x27;by&#x27;,
                                                &#x27;can&#x27;, &#x27;couldn&#x27;, &quot;couldn&#x27;t&quot;, ...],
                                               None],
                          &#x27;vect__tokenizer&#x27;: [&lt;function tokenizer at 0x7fe3cf04ec20&gt;],
                          &#x27;vect__use_idf&#x27;: [False]}],
             scoring=&#x27;accuracy&#x27;, verbose=2)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;vect&#x27;, TfidfVectorizer(lowercase=False)),
                (&#x27;clf&#x27;,
                 LogisticRegression(random_state=0, solver=&#x27;liblinear&#x27;))])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label sk-toggleable__label-arrow">TfidfVectorizer</label><div class="sk-toggleable__content"><pre>TfidfVectorizer(lowercase=False)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" ><label for="sk-estimator-id-8" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(random_state=0, solver=&#x27;liblinear&#x27;)</pre></div></div></div></div></div></div></div></div></div></div></div></div>



    [CV] END clf__C=1.0, clf__penalty=l2, vect__ngram_range=(1, 1), vect__norm=None, vect__tokenizer=<function tokenizer at 0x7fcb76cb8940>, vect__use_idf=True; total time=  28.1s
    [CV] END clf__C=10.0, clf__penalty=l1, vect__ngram_range=(1, 1), vect__norm=None, vect__tokenizer=<function tokenizer at 0x7fcb76cb8af0>, vect__use_idf=True; total time=  26.9s
    [CV] END clf__C=1.0, clf__penalty=l1, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', "he'd", "he'll", 'her', 'here', 'hers', 'herself', "he's", 'him', 'himself', 'his', 'how', 'i', "i'd", 'if', "i'll", "i'm", 'in', 'into', 'is', 'isn', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', "i've", 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she'd", "she'll", "she's", 'should', 'shouldn', "shouldn't", "should've", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', "we'd", "we'll", "we're", 'were', 'weren', "weren't", "we've", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", 'your', "you're", 'yours', 'yourself', 'yourselves', "you've"], vect__tokenizer=<function tokenizer at 0x7fcb76cb8c10>, vect__use_idf=False; total time=  27.1s
    [CV] END clf__C=1.0, clf__penalty=l1, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=None, vect__tokenizer=<function tokenizer at 0x7fcb76cb8af0>, vect__use_idf=False; total time=  27.2s
    [CV] END clf__C=1.0, clf__penalty=l2, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', "he'd", "he'll", 'her', 'here', 'hers', 'herself', "he's", 'him', 'himself', 'his', 'how', 'i', "i'd", 'if', "i'll", "i'm", 'in', 'into', 'is', 'isn', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', "i've", 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she'd", "she'll", "she's", 'should', 'shouldn', "shouldn't", "should've", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', "we'd", "we'll", "we're", 'were', 'weren', "weren't", "we've", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", 'your', "you're", 'yours', 'yourself', 'yourselves', "you've"], vect__tokenizer=<function tokenizer at 0x7fcb76cb8c10>, vect__use_idf=False; total time=  27.1s
    [CV] END clf__C=10.0, clf__penalty=l1, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', "he'd", "he'll", 'her', 'here', 'hers', 'herself', "he's", 'him', 'himself', 'his', 'how', 'i', "i'd", 'if', "i'll", "i'm", 'in', 'into', 'is', 'isn', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', "i've", 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she'd", "she'll", "she's", 'should', 'shouldn', "shouldn't", "should've", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', "we'd", "we'll", "we're", 'were', 'weren', "weren't", "we've", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", 'your', "you're", 'yours', 'yourself', 'yourselves', "you've"], vect__tokenizer=<function tokenizer at 0x7fcb76cb8af0>, vect__use_idf=False; total time=  27.5s
    [CV] END clf__C=10.0, clf__penalty=l2, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', "he'd", "he'll", 'her', 'here', 'hers', 'herself', "he's", 'him', 'himself', 'his', 'how', 'i', "i'd", 'if', "i'll", "i'm", 'in', 'into', 'is', 'isn', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', "i've", 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she'd", "she'll", "she's", 'should', 'shouldn', "shouldn't", "should've", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', "we'd", "we'll", "we're", 'were', 'weren', "weren't", "we've", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", 'your', "you're", 'yours', 'yourself', 'yourselves', "you've"], vect__tokenizer=<function tokenizer at 0x7fcb76cb8c10>, vect__use_idf=False; total time=  27.6s
    [CV] END clf__C=1.0, clf__penalty=l1, vect__ngram_range=(1, 1), vect__norm=None, vect__tokenizer=<function tokenizer at 0x7fc46acb4940>, vect__use_idf=True; total time=  25.9s
    [CV] END clf__C=1.0, clf__penalty=l2, vect__ngram_range=(1, 1), vect__norm=None, vect__tokenizer=<function tokenizer at 0x7fc46acb4af0>, vect__use_idf=True; total time=  28.6s
    [CV] END clf__C=10.0, clf__penalty=l2, vect__ngram_range=(1, 1), vect__norm=None, vect__tokenizer=<function tokenizer at 0x7fc46acb4c10>, vect__use_idf=True; total time=  29.4s
    [CV] END clf__C=1.0, clf__penalty=l2, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', "he'd", "he'll", 'her', 'here', 'hers', 'herself', "he's", 'him', 'himself', 'his', 'how', 'i', "i'd", 'if', "i'll", "i'm", 'in', 'into', 'is', 'isn', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', "i've", 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she'd", "she'll", "she's", 'should', 'shouldn', "shouldn't", "should've", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', "we'd", "we'll", "we're", 'were', 'weren', "weren't", "we've", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", 'your', "you're", 'yours', 'yourself', 'yourselves', "you've"], vect__tokenizer=<function tokenizer at 0x7fc46acb4af0>, vect__use_idf=False; total time=  27.6s
    [CV] END clf__C=1.0, clf__penalty=l2, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=None, vect__tokenizer=<function tokenizer at 0x7fc46acb4c10>, vect__use_idf=False; total time=  26.3s
    [CV] END clf__C=10.0, clf__penalty=l1, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=None, vect__tokenizer=<function tokenizer at 0x7fc46acb4af0>, vect__use_idf=False; total time=  26.6s
    [CV] END clf__C=10.0, clf__penalty=l2, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', "he'd", "he'll", 'her', 'here', 'hers', 'herself', "he's", 'him', 'himself', 'his', 'how', 'i', "i'd", 'if', "i'll", "i'm", 'in', 'into', 'is', 'isn', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', "i've", 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she'd", "she'll", "she's", 'should', 'shouldn', "shouldn't", "should've", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', "we'd", "we'll", "we're", 'were', 'weren', "weren't", "we've", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", 'your', "you're", 'yours', 'yourself', 'yourselves', "you've"], vect__tokenizer=<function tokenizer at 0x7fc46acb4c10>, vect__use_idf=False; total time=  27.5s


    [CV] END clf__C=1.0, clf__penalty=l1, vect__ngram_range=(1, 1), vect__norm=None, vect__tokenizer=<function tokenizer at 0x7fe9ca4b8940>, vect__use_idf=True; total time=  26.1s
    [CV] END clf__C=10.0, clf__penalty=l1, vect__ngram_range=(1, 1), vect__norm=None, vect__tokenizer=<function tokenizer at 0x7fe9ca4b8af0>, vect__use_idf=True; total time=  26.9s
    [CV] END clf__C=10.0, clf__penalty=l2, vect__ngram_range=(1, 1), vect__norm=None, vect__tokenizer=<function tokenizer at 0x7fe9ca4b8c10>, vect__use_idf=True; total time=  29.8s
    [CV] END clf__C=1.0, clf__penalty=l1, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=None, vect__tokenizer=<function tokenizer at 0x7fe9ca4b8af0>, vect__use_idf=False; total time=  27.1s
    [CV] END clf__C=1.0, clf__penalty=l2, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=None, vect__tokenizer=<function tokenizer at 0x7fe9ca4b8c10>, vect__use_idf=False; total time=  26.9s
    [CV] END clf__C=10.0, clf__penalty=l1, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', "he'd", "he'll", 'her', 'here', 'hers', 'herself', "he's", 'him', 'himself', 'his', 'how', 'i', "i'd", 'if', "i'll", "i'm", 'in', 'into', 'is', 'isn', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', "i've", 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she'd", "she'll", "she's", 'should', 'shouldn', "shouldn't", "should've", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', "we'd", "we'll", "we're", 'were', 'weren', "weren't", "we've", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", 'your', "you're", 'yours', 'yourself', 'yourselves', "you've"], vect__tokenizer=<function tokenizer at 0x7fe9ca4b8af0>, vect__use_idf=False; total time=  28.0s
    [CV] END clf__C=10.0, clf__penalty=l2, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', "he'd", "he'll", 'her', 'here', 'hers', 'herself', "he's", 'him', 'himself', 'his', 'how', 'i', "i'd", 'if', "i'll", "i'm", 'in', 'into', 'is', 'isn', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', "i've", 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she'd", "she'll", "she's", 'should', 'shouldn', "shouldn't", "should've", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', "we'd", "we'll", "we're", 'were', 'weren', "weren't", "we've", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", 'your', "you're", 'yours', 'yourself', 'yourselves', "you've"], vect__tokenizer=<function tokenizer at 0x7fe9ca4b8c10>, vect__use_idf=False; total time=  27.7s
    [CV] END clf__C=1.0, clf__penalty=l2, vect__ngram_range=(1, 1), vect__norm=None, vect__tokenizer=<function tokenizer at 0x7fd798cbc940>, vect__use_idf=True; total time=  28.1s
    [CV] END clf__C=10.0, clf__penalty=l2, vect__ngram_range=(1, 1), vect__norm=None, vect__tokenizer=<function tokenizer at 0x7fd798cbcaf0>, vect__use_idf=True; total time=  29.0s
    [CV] END clf__C=1.0, clf__penalty=l1, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', "he'd", "he'll", 'her', 'here', 'hers', 'herself', "he's", 'him', 'himself', 'his', 'how', 'i', "i'd", 'if', "i'll", "i'm", 'in', 'into', 'is', 'isn', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', "i've", 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she'd", "she'll", "she's", 'should', 'shouldn', "shouldn't", "should've", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', "we'd", "we'll", "we're", 'were', 'weren', "weren't", "we've", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", 'your', "you're", 'yours', 'yourself', 'yourselves', "you've"], vect__tokenizer=<function tokenizer at 0x7fd798cbcc10>, vect__use_idf=False; total time=  26.8s
    [CV] END clf__C=1.0, clf__penalty=l2, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', "he'd", "he'll", 'her', 'here', 'hers', 'herself', "he's", 'him', 'himself', 'his', 'how', 'i', "i'd", 'if', "i'll", "i'm", 'in', 'into', 'is', 'isn', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', "i've", 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she'd", "she'll", "she's", 'should', 'shouldn', "shouldn't", "should've", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', "we'd", "we'll", "we're", 'were', 'weren', "weren't", "we've", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", 'your', "you're", 'yours', 'yourself', 'yourselves', "you've"], vect__tokenizer=<function tokenizer at 0x7fd798cbcaf0>, vect__use_idf=False; total time=  27.6s
    [CV] END clf__C=1.0, clf__penalty=l2, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=None, vect__tokenizer=<function tokenizer at 0x7fd798cbcc10>, vect__use_idf=False; total time=  26.4s
    [CV] END clf__C=10.0, clf__penalty=l1, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=None, vect__tokenizer=<function tokenizer at 0x7fd798cbcaf0>, vect__use_idf=False; total time=  27.5s
    [CV] END clf__C=10.0, clf__penalty=l2, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=None, vect__tokenizer=<function tokenizer at 0x7fd798cbcc10>, vect__use_idf=False; total time=  26.9s


    [CV] END clf__C=1.0, clf__penalty=l1, vect__ngram_range=(1, 1), vect__norm=None, vect__tokenizer=<function tokenizer at 0x7fd5544b8940>, vect__use_idf=True; total time=  26.0s
    [CV] END clf__C=1.0, clf__penalty=l2, vect__ngram_range=(1, 1), vect__norm=None, vect__tokenizer=<function tokenizer at 0x7fd5544b8af0>, vect__use_idf=True; total time=  28.5s
    [CV] END clf__C=1.0, clf__penalty=l1, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', "he'd", "he'll", 'her', 'here', 'hers', 'herself', "he's", 'him', 'himself', 'his', 'how', 'i', "i'd", 'if', "i'll", "i'm", 'in', 'into', 'is', 'isn', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', "i've", 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she'd", "she'll", "she's", 'should', 'shouldn', "shouldn't", "should've", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', "we'd", "we'll", "we're", 'were', 'weren', "weren't", "we've", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", 'your', "you're", 'yours', 'yourself', 'yourselves', "you've"], vect__tokenizer=<function tokenizer at 0x7fd5544b8c10>, vect__use_idf=False; total time=  27.1s
    [CV] END clf__C=1.0, clf__penalty=l1, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', "he'd", "he'll", 'her', 'here', 'hers', 'herself', "he's", 'him', 'himself', 'his', 'how', 'i', "i'd", 'if', "i'll", "i'm", 'in', 'into', 'is', 'isn', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', "i've", 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she'd", "she'll", "she's", 'should', 'shouldn', "shouldn't", "should've", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', "we'd", "we'll", "we're", 'were', 'weren', "weren't", "we've", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", 'your', "you're", 'yours', 'yourself', 'yourselves', "you've"], vect__tokenizer=<function tokenizer at 0x7fd5544b8af0>, vect__use_idf=False; total time=  27.2s
    [CV] END clf__C=1.0, clf__penalty=l2, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', "he'd", "he'll", 'her', 'here', 'hers', 'herself', "he's", 'him', 'himself', 'his', 'how', 'i', "i'd", 'if', "i'll", "i'm", 'in', 'into', 'is', 'isn', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', "i've", 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she'd", "she'll", "she's", 'should', 'shouldn', "shouldn't", "should've", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', "we'd", "we'll", "we're", 'were', 'weren', "weren't", "we've", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", 'your', "you're", 'yours', 'yourself', 'yourselves', "you've"], vect__tokenizer=<function tokenizer at 0x7fd5544b8c10>, vect__use_idf=False; total time=  26.9s
    [CV] END clf__C=10.0, clf__penalty=l1, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', "he'd", "he'll", 'her', 'here', 'hers', 'herself', "he's", 'him', 'himself', 'his', 'how', 'i', "i'd", 'if', "i'll", "i'm", 'in', 'into', 'is', 'isn', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', "i've", 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she'd", "she'll", "she's", 'should', 'shouldn', "shouldn't", "should've", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', "we'd", "we'll", "we're", 'were', 'weren', "weren't", "we've", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", 'your', "you're", 'yours', 'yourself', 'yourselves', "you've"], vect__tokenizer=<function tokenizer at 0x7fd5544b8af0>, vect__use_idf=False; total time=  27.1s
    [CV] END clf__C=10.0, clf__penalty=l1, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=None, vect__tokenizer=<function tokenizer at 0x7fd5544b8c10>, vect__use_idf=False; total time=  26.8s
    [CV] END clf__C=10.0, clf__penalty=l2, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=None, vect__tokenizer=<function tokenizer at 0x7fd5544b8af0>, vect__use_idf=False; total time=  15.3s
    [CV] END clf__C=1.0, clf__penalty=l2, vect__ngram_range=(1, 1), vect__norm=None, vect__tokenizer=<function tokenizer at 0x7fa42ccb4940>, vect__use_idf=True; total time=  28.0s
    [CV] END clf__C=10.0, clf__penalty=l1, vect__ngram_range=(1, 1), vect__norm=None, vect__tokenizer=<function tokenizer at 0x7fa42ccb4af0>, vect__use_idf=True; total time=  26.7s
    [CV] END clf__C=1.0, clf__penalty=l1, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', "he'd", "he'll", 'her', 'here', 'hers', 'herself', "he's", 'him', 'himself', 'his', 'how', 'i', "i'd", 'if', "i'll", "i'm", 'in', 'into', 'is', 'isn', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', "i've", 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she'd", "she'll", "she's", 'should', 'shouldn', "shouldn't", "should've", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', "we'd", "we'll", "we're", 'were', 'weren', "weren't", "we've", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", 'your', "you're", 'yours', 'yourself', 'yourselves', "you've"], vect__tokenizer=<function tokenizer at 0x7fa42ccb4c10>, vect__use_idf=False; total time=  27.4s
    [CV] END clf__C=1.0, clf__penalty=l1, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=None, vect__tokenizer=<function tokenizer at 0x7fa42ccb4af0>, vect__use_idf=False; total time=  27.2s
    [CV] END clf__C=1.0, clf__penalty=l2, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', "he'd", "he'll", 'her', 'here', 'hers', 'herself', "he's", 'him', 'himself', 'his', 'how', 'i', "i'd", 'if', "i'll", "i'm", 'in', 'into', 'is', 'isn', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', "i've", 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she'd", "she'll", "she's", 'should', 'shouldn', "shouldn't", "should've", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', "we'd", "we'll", "we're", 'were', 'weren', "weren't", "we've", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", 'your', "you're", 'yours', 'yourself', 'yourselves', "you've"], vect__tokenizer=<function tokenizer at 0x7fa42ccb4c10>, vect__use_idf=False; total time=  27.1s
    [CV] END clf__C=10.0, clf__penalty=l1, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', "he'd", "he'll", 'her', 'here', 'hers', 'herself', "he's", 'him', 'himself', 'his', 'how', 'i', "i'd", 'if', "i'll", "i'm", 'in', 'into', 'is', 'isn', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', "i've", 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she'd", "she'll", "she's", 'should', 'shouldn', "shouldn't", "should've", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', "we'd", "we'll", "we're", 'were', 'weren', "weren't", "we've", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", 'your', "you're", 'yours', 'yourself', 'yourselves', "you've"], vect__tokenizer=<function tokenizer at 0x7fa42ccb4af0>, vect__use_idf=False; total time=  26.7s
    [CV] END clf__C=10.0, clf__penalty=l1, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=None, vect__tokenizer=<function tokenizer at 0x7fa42ccb4c10>, vect__use_idf=False; total time=  26.8s
    [CV] END clf__C=10.0, clf__penalty=l2, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=None, vect__tokenizer=<function tokenizer at 0x7fa42ccb4af0>, vect__use_idf=False; total time=  15.1s


    [CV] END clf__C=1.0, clf__penalty=l1, vect__ngram_range=(1, 1), vect__norm=None, vect__tokenizer=<function tokenizer at 0x7f9543cb4940>, vect__use_idf=True; total time=  26.1s
    [CV] END clf__C=10.0, clf__penalty=l1, vect__ngram_range=(1, 1), vect__norm=None, vect__tokenizer=<function tokenizer at 0x7f9543cb4af0>, vect__use_idf=True; total time=  27.1s
    [CV] END clf__C=10.0, clf__penalty=l2, vect__ngram_range=(1, 1), vect__norm=None, vect__tokenizer=<function tokenizer at 0x7f9543cb4c10>, vect__use_idf=True; total time=  29.9s
    [CV] END clf__C=1.0, clf__penalty=l1, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=None, vect__tokenizer=<function tokenizer at 0x7f9543cb4af0>, vect__use_idf=False; total time=  27.1s
    [CV] END clf__C=1.0, clf__penalty=l2, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=None, vect__tokenizer=<function tokenizer at 0x7f9543cb4c10>, vect__use_idf=False; total time=  26.6s
    [CV] END clf__C=10.0, clf__penalty=l1, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', "he'd", "he'll", 'her', 'here', 'hers', 'herself', "he's", 'him', 'himself', 'his', 'how', 'i', "i'd", 'if', "i'll", "i'm", 'in', 'into', 'is', 'isn', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', "i've", 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she'd", "she'll", "she's", 'should', 'shouldn', "shouldn't", "should've", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', "we'd", "we'll", "we're", 'were', 'weren', "weren't", "we've", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", 'your', "you're", 'yours', 'yourself', 'yourselves', "you've"], vect__tokenizer=<function tokenizer at 0x7f9543cb4af0>, vect__use_idf=False; total time=  27.3s
    [CV] END clf__C=10.0, clf__penalty=l2, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', "he'd", "he'll", 'her', 'here', 'hers', 'herself', "he's", 'him', 'himself', 'his', 'how', 'i', "i'd", 'if', "i'll", "i'm", 'in', 'into', 'is', 'isn', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', "i've", 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she'd", "she'll", "she's", 'should', 'shouldn', "shouldn't", "should've", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', "we'd", "we'll", "we're", 'were', 'weren', "weren't", "we've", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", 'your', "you're", 'yours', 'yourself', 'yourselves', "you've"], vect__tokenizer=<function tokenizer at 0x7f9543cb4c10>, vect__use_idf=False; total time=  27.4s
    [CV] END clf__C=10.0, clf__penalty=l2, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=None, vect__tokenizer=<function tokenizer at 0x7f9543cb4af0>, vect__use_idf=False; total time=  14.3s
    [CV] END clf__C=1.0, clf__penalty=l1, vect__ngram_range=(1, 1), vect__norm=None, vect__tokenizer=<function tokenizer at 0x7fbff6e20940>, vect__use_idf=True; total time=  26.2s
    [CV] END clf__C=10.0, clf__penalty=l1, vect__ngram_range=(1, 1), vect__norm=None, vect__tokenizer=<function tokenizer at 0x7fbff6e20af0>, vect__use_idf=True; total time=  26.9s
    [CV] END clf__C=10.0, clf__penalty=l2, vect__ngram_range=(1, 1), vect__norm=None, vect__tokenizer=<function tokenizer at 0x7fbff6e20c10>, vect__use_idf=True; total time=  29.6s
    [CV] END clf__C=1.0, clf__penalty=l1, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=None, vect__tokenizer=<function tokenizer at 0x7fbff6e20af0>, vect__use_idf=False; total time=  27.2s
    [CV] END clf__C=1.0, clf__penalty=l2, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=None, vect__tokenizer=<function tokenizer at 0x7fbff6e20c10>, vect__use_idf=False; total time=  26.7s
    [CV] END clf__C=10.0, clf__penalty=l1, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=None, vect__tokenizer=<function tokenizer at 0x7fbff6e20af0>, vect__use_idf=False; total time=  27.3s
    [CV] END clf__C=10.0, clf__penalty=l2, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', "he'd", "he'll", 'her', 'here', 'hers', 'herself', "he's", 'him', 'himself', 'his', 'how', 'i', "i'd", 'if', "i'll", "i'm", 'in', 'into', 'is', 'isn', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', "i've", 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she'd", "she'll", "she's", 'should', 'shouldn', "shouldn't", "should've", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', "we'd", "we'll", "we're", 'were', 'weren', "weren't", "we've", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", 'your', "you're", 'yours', 'yourself', 'yourselves', "you've"], vect__tokenizer=<function tokenizer at 0x7fbff6e20c10>, vect__use_idf=False; total time=  27.6s
    [CV] END clf__C=10.0, clf__penalty=l2, vect__ngram_range=(1, 1), vect__norm=l2, vect__stop_words=None, vect__tokenizer=<function tokenizer at 0x7fbff6e20af0>, vect__use_idf=False; total time=  14.2s



```python
print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)
```

    Best parameter set: {'clf__C': 10.0, 'clf__penalty': 'l2', 'vect__ngram_range': (1, 1), 'vect__norm': 'l2', 'vect__stop_words': ['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', "he'd", "he'll", 'her', 'here', 'hers', 'herself', "he's", 'him', 'himself', 'his', 'how', 'i', "i'd", 'if', "i'll", "i'm", 'in', 'into', 'is', 'isn', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', "i've", 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she'd", "she'll", "she's", 'should', 'shouldn', "shouldn't", "should've", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', "we'd", "we'll", "we're", 'were', 'weren', "weren't", "we've", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", 'your', "you're", 'yours', 'yourself', 'yourselves', "you've"], 'vect__tokenizer': <function tokenizer at 0x7fe3cf04ec20>, 'vect__use_idf': False} 
    CV Accuracy: 0.884



```python
clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))
```

    Test Accuracy: 0.892


## Topic modeling with Latent Dirichlet Allocation (LDA)

- topic modeling describes the broad task of assigning topics to unlabeled text documents
- e.g., automatic categorization of documents in a large text corpus of newspaper articles into topics:
    - sports, finance, world news, politics, local news, etc.
- topic modeling is a type of clustering task (a subcategory of unsupervised learning)
- let's use `LatentDirichletAllocation` class implemented in scikit-learn to learn different topics from the IMDb movie dataset


```python
import pandas as pd
import pickle
```


```python
# load the pickle dump
df = pickle.load(open('./data/movie_data.pd', 'rb'))
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11841</th>
      <td>In 1974, the teenager Martha Moxley (Maggie Gr...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19602</th>
      <td>OK... so... I really like Kris Kristofferson a...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>45519</th>
      <td>***SPOILER*** Do not read this, if you think a...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25747</th>
      <td>hi for all the people who have seen this wonde...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>42642</th>
      <td>I recently bought the DVD, forgetting just how...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21243</th>
      <td>OK, lets start with the best. the building. al...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>45891</th>
      <td>The British 'heritage film' industry is out of...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>42613</th>
      <td>I don't even know where to begin on this one. ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>43567</th>
      <td>Richard Tyler is a little boy who is scared of...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2732</th>
      <td>I waited long to watch this movie. Also becaus...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>50000 rows × 2 columns</p>
</div>




```python
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english',
                        max_df=.1,
                        max_features=5_000)
X = count.fit_transform(df['review'].values)
# hyperparameters: max_df = 10% - to exclude words that occur too frequently across documents
# limit the max features to 5000; limit dimensionality of the dataset
```


```python
# Note this may take a while... about 5 mins
from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=10, # topics
                                random_state=123,
                                learning_method='batch')
# batch learning method is slower compared to 'online' but may lead to better accuracy
X_topics = lda.fit_transform(X)
```


```python
lda.components_.shape
```




    (10, 5000)




```python
# let's print the 5 most important words for each of the 10 topics
n_top_words = 5
feature_names = count.get_feature_names()

for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx + 1))
    print(" ".join([feature_names[i]
                    for i in topic.argsort()\
                        [:-n_top_words - 1:-1]]))
```

    Topic 1:
    worst minutes awful script stupid
    Topic 2:
    family mother father children girl
    Topic 3:
    american war dvd music tv
    Topic 4:
    human audience cinema art sense
    Topic 5:
    police guy car dead murder
    Topic 6:
    horror house sex girl woman
    Topic 7:
    role performance comedy actor performances
    Topic 8:
    series episode war episodes tv
    Topic 9:
    book version original read novel
    Topic 10:
    action fight guy guys cool


    /opt/anaconda3/envs/ml/lib/python3.10/site-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function get_feature_names is deprecated; get_feature_names is deprecated in 1.0 and will be removed in 1.2. Please use get_feature_names_out instead.
      warnings.warn(msg, category=FutureWarning)


- based on reading the 5 most important words for each topic, we can guess that the LDA identified the following topics:
    
1. Generally bad movies (not really a topic category)
2. Movies about families
3. War movies
4. Art movies
5. Crime movies
6. Horror movies
7. Comedies
8. Movies somehow related to TV shows
9. Movies based on books
10. Action movies

- let's look at the actual contents of the reviews
- print 5 movies from the each category


```python
topic_index = 2
horror = X_topics[:, topic_index].argsort()[::-1]

for iter_idx, movie_idx in enumerate(horror[:5]):
    print(f'\nTopic {topic_index} - movide #{iter_idx+1}')
    print(df['review'][movie_idx][:500], '...')
```

    
    Topic 2 - movide #1
    ***THIS POST CONTAINS SPOILER(S)*** <br /><br />I'm not a big fan of Chuck Norris as an actor, but I worship him in all other ways. He also have his own fan web site with "Chuck Norris facts" that is really entertaining. But this movie looks like someone was joking with the audience putting all those "facts" into one movie. I really don't remember when I wasted my time more than with this "action". I don't know what's the worst this movie can offer you: unoriginal and thousand-times-made plot of ...
    
    Topic 2 - movide #2
    I first saw this when I was around 7. I remembered what I believed to be a vague outline of what took place. Turns out now, 15 years later, that I remembered everything with great accuracy because it seems the writers never got beyond making an outline to the story. There is no plot to this movie/cartoon. There is no character development, no back story, no character arcs, nothing. The good guys do things because they are good, while the bad guys do things solely because they are bad. One uninte ...
    
    Topic 2 - movide #3
    Written by someone who has been there, you can tell, but only if you've been there. Excellent performances by Meryl Streep (of course!), Renee Zellweger and William Hurt.<br /><br />Many people have said that it is about a dysfunctional family, I think every family is dysfunctional when they are facing this kind of torment. To NOT be dysfunctional would be dysfunctional! You are losing your family as you know it, can anything be worse? People need to see this movie so when they are faced with th ...
    
    Topic 2 - movide #4
    This PM Entertainment production is laced with enough bullets to make John Woo say, "Enough already!" Of course, it isn't nearly as beautiful as Woo can deliver but it gets the exploitive job done in 90 minutes. Eric Phillips (Don Wilson) is an undercover cop in the near future. When his wife is framed for murdering the Governor by a team using a look-a-like cyborg, it is up to Eric to clear her name. Wilson gets to pull Van Damme duty as he plays the heroic lead and his evil cyborg doppelganger ...
    
    Topic 2 - movide #5
    "The Mother" tells of a recently widowed mid-60's mother of two adult children (Reid) who, on the heels of her husband's death, finds herself awakening from a life of sleepwalking as she has an affair with a young carpenter who is also her daughter's married lover. The film dwells on the quietly passive Mom, her tenuous relationship with her grown son and daughter, the silent needs she attempts to soothe in bed with her young lover, and the convolutions arising therefrom. A somewhat antiseptic d ...





```python

```
