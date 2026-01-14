# Model Deployment
- ML techniques are not limited to offline application and analyses
- they have become predictive engine of various web services
    - spam detection, search engines, recommendation systems, etc.
    - online demo CNN for digit recognition: https://www.denseinl2.com/webcnn/digitdemo.html 
    - nice live training demo with visualization: https://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html
- the goal of this chapter is to learn how to deploy a trained model and use it to classify new samples and also continuously learn from data in real time

## Working with bigger data
- it's normal to have hundreds of thousands of samples in dataset e.g. in text classification problems
- in the era of big data (terabyes and petabytes), it's not uncommon to have dataset that doesn't fit in the desktop computer memory
- either employ supercomputers or apply **out-of-core learning** with online algorithms
- see https://scikit-learn.org/0.15/modules/scaling_strategies.html

### out-of-core learning
- allows us to work with large datasets by fitting the classifier incrementally on smaller batches of a dataset
    
### online algorithms
- algorithms that don't need all the training samples at once but can be trained in batches over time
    - also called incremental algorithms
- these algorithms have `partial_fit` method in sci-kit learn framework
- use **stochastic gradient descent** optimization algorithm that updates the models's weights using one example at a time
- let's use `partial_fit` method of incremental SGDClassifier to train a logistric regression model using small mini-batches of documents

#### Key Aspects of SGD algorithm

- Unlike Gradient Descent that uses whole dataset to update the weights, Stochastic GD can use one or smaller batch of training samples

1. **Initialization**: 
    - starts with an initial random guess for the model's weights
   
2. **Iteration (Epochs and Steps)**:
    - training data is often shuffled
    - the algorithm iterates through the training data (or a number of epochs)
    - in each iteration (or step), a single data point (or a mini-batch) is randomly selected
    - the gradient of the loss function is calculated with respect to the model's parameters using only a single sample (or mini-batch)
    - model's parameters are updated in the opposite direction of this gradient, scaled by a learning rate (a hyperparameter that controls the step size)
    
3. **Learning Rate**:
    - learning rate is crucial
    - a large learning rate might cause the algorithm to overshoot the minimum, while a small learning rate might lead to slow convergence
    
4. **Stopping Criteria**:
    - the algorithm stops when a certain number of iterations (epochs) is reached, when the change in the loss function is below a threshold, or by other criteria

## Use movie_data.csv file

- we'll deploy sentiment analysis of movie dataset analyzed in the Sentiment Analysis chapter
- use code provided there to create CSV file from dataframe
- the following code checks for `data/movie_data.csv` file a


```python
import gzip
from pathlib import Path
# check if file exists otherwise download and unzip the zipped imdb dataset
file = Path('./data')/'movie_data.csv'
if file.exists():
    print(f'File {file} exists! Continue...')
else:
    print(f'{file} does not exist!')
    print('Please place a copy of the movie_data.csv.gz'
          'in this directory. You can obtain it by'
          'a) executing the code in the previous'
          'notebook or b) by downloading it from GitHub:'
          'https://github.com/rasbt/python-machine-learning-'
          'book-2nd-edition/blob/master/code/ch08/movie_data.csv.gz')
```

    File data/movie_data.csv exists! Continue...



```python
import numpy as np
import re
from nltk.corpus import stopwords

stop = stopwords.words('english')


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

# create an iterator function to yield text and label
def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)  # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label
```


```python
# use next function to get the next document from the iterator
next(stream_docs(path=file))
# should return a tuple of (text, label)
```




    ('"In 1974, the teenager Martha Moxley (Maggie Grace) moves to the high-class area of Belle Haven, Greenwich, Connecticut. On the Mischief Night, eve of Halloween, she was murdered in the backyard of her house and her murder remained unsolved. Twenty-two years later, the writer Mark Fuhrman (Christopher Meloni), who is a former LA detective that has fallen in disgrace for perjury in O.J. Simpson trial and moved to Idaho, decides to investigate the case with his partner Stephen Weeks (Andrew Mitchell) with the purpose of writing a book. The locals squirm and do not welcome them, but with the support of the retired detective Steve Carroll (Robert Forster) that was in charge of the investigation in the 70\'s, they discover the criminal and a net of power and money to cover the murder.<br /><br />""Murder in Greenwich"" is a good TV movie, with the true story of a murder of a fifteen years old girl that was committed by a wealthy teenager whose mother was a Kennedy. The powerful and rich family used their influence to cover the murder for more than twenty years. However, a snoopy detective and convicted perjurer in disgrace was able to disclose how the hideous crime was committed. The screenplay shows the investigation of Mark and the last days of Martha in parallel, but there is a lack of the emotion in the dramatization. My vote is seven.<br /><br />Title (Brazil): Not Available"',
     1)




```python
# function takes stream_docs function and return a number of documents specified by size
def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y
```

### HashingVectorizer
- can't use `CountVectorizer` and `TfidfVectorizer` for out-of-core learning
    - they require holding the complete vocabulary and documents in memory
- `HashingVectorizer` is data-independent and makes use of the hashing trick via 32-bit MurmurShash3 algorithm
- difference between `CountVectorizer` and `HashingVectorizer`: https://kavita-ganesan.com/hashingvectorizer-vs-countvectorizer/#.YFF_lbRKhTY


```python
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

# create HashingVectorizer object with 2**21 max slots
vect = HashingVectorizer(decode_error='ignore', 
                         n_features=2**21,
                         preprocessor=None, 
                         tokenizer=tokenizer)
```


```python
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version

clf = SGDClassifier(loss='log_loss', random_state=1)

doc_stream = stream_docs(path=file)
```


```python
# let's train the model in batch; display the status with pyprind library
# takes about 20 seconds
import pyprind
pbar = pyprind.ProgBar(45)

classes = np.array([0, 1])
# use 45 mini batches each with 1_000 documents
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1_000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()
```

    0% [##############################] 100% | ETA: 00:00:00
    Total time elapsed: 00:00:23



```python
X_test, y_test = get_minibatch(stream_docs(path=file), 10_000)
```


```python
# let's use the last 5000 samples to test our model
#X_test, y_test = get_minibatch(doc_stream, size=5_000)
X_test = vect.transform(X_test)
print('Accuracy: %.3f' % clf.score(X_test, y_test))
```

    Accuracy: 0.879


### Note:
- eventhough the accuracy is slightly lower compared to offline learning with grid search technique, training time is much faster!
- we can incrementally train the model with more data
    - let's use the 5000 test samples we've not used to train the model yet


```python
clf = clf.partial_fit(X_test, y_test)
```


```python
# lets test the model again out of curiosity
print('Accuracy: %.3f' % clf.score(X_test, y_test))
# accuracy went up by about 2%
```

    Accuracy: 0.888


## Serializing fitted scikit-learn estimators
- training a machine learning algorithm can be computationally expensive
- don't want to retrain our model every time we close our Python interpreter and want to make a new prediction or reload our web application
- one option is to use Python's `pickle` module
    - `pickle` can serilaize and deserialize Python object structures to compact bytecode
    - save our classifier in its current state and reload it when we want to classify new, unlabeled examples


```python
import pickle
import os
dest = './demos/movieclassifier/pkl_objects'
if not os.path.exists(dest):
    os.makedirs(dest)
# let's serialize the stop-word set
pickle.dump(stop, open(os.path.join(dest, 'stopwords.pkl'), 'wb'), protocol=4)
# let's serialize the trained classifier
pickle.dump(clf,
    open(os.path.join(dest, 'classifier.pkl'), 'wb'), protocol=4)
```


```python
%%writefile ./demos/movieclassifier/vectorizer.py
# the above Jupyter notebook magic writes the code in the cell to the provided file; must be the first line!
from sklearn.feature_extraction.text import HashingVectorizer
import re
import os
import pickle

cur_dir = os.path.dirname(__file__)
stop = pickle.load(open(
                os.path.join(cur_dir, 
                'pkl_objects', 
                'stopwords.pkl'), 'rb'))

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) \
                   + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)
```

    Writing ./demos/movieclassifier/vectorizer.py



```python
# let's deserialize the pickle objects and test them
# change the current working directory to demos/movieclassifier
#import os
#os.chdir('demos/movieclassifier')
%cd demos/movieclassifier
```

    /Users/rbasnet/projects/python-machine-learning/notebooks/demos/movieclassifier



```python
! pwd
```

    /Users/rbasnet/projects/python-machine-learning/notebooks/demos/movieclassifier



```python
# deserialize the classifer
import pickle
import re
import os
# this is our module generated in above cell
from vectorizer import vect
import numpy as np

clf_pk = pickle.load(open(os.path.join('pkl_objects', 'classifier.pkl'), 'rb'))
```


```python
def result(label, prob):
    if label == 1:
        if prob >= 90:
            return ':D'
        elif prob >= 70:
            return ':)'
        else:
            return ':|'
    else:
        if prob >= 90:
            return ':`('
        elif prob >= 70:
            return ':('
        else:
            return ':|'
```


```python
# let's test the classifier with some reviews
label = {0:'negative', 1:'positive'}

example = ["I love this movie. It's amazing."]
X = vect.transform(example)
# predict returns the class label with the largest probability
lbl = clf.predict(X)
# predict_prob method returns the probability estimate for the sample
prob = np.max(clf.predict_proba(X))*100
print('Prediction: %s\nProbability: %.2f%%' %\
      (label[lbl[0]], prob))
print('Result: ', result(lbl, prob))
```

    Prediction: positive
    Probability: 96.36%
    Result:  :D



```python
# let's create a function so we can reuse the code easily...
def predict(review):
    label = {0:'negative', 1:'positive'}
    example = [review]
    X = vect.transform(example)
    # predict returns the class label with the largest probability
    lbl = clf.predict(X)
    # predict_prob method returns the probability estimate for the sample
    prob = np.max(clf.predict_proba(X))*100
    print('Prediction: %s\nProbability: %.2f%%' %\
          (label[lbl[0]], prob))
    print('Result: ', result(lbl, prob))
```


```python
predict("The movie was so boring that I slept through it!")
```

    Prediction: negative
    Probability: 94.09%
    Result:  :`(



```python
predict("The movie was okay but I'd not watch it again!")
```

    Prediction: negative
    Probability: 64.79%
    Result:  :|



```python
review = input('Enter your review: ')
```

    Enter your review: the best movie I've ever seen!



```python
predict(review)
```

    Prediction: positive
    Probability: 89.19%
    Result:  :)


## Web application with Flask
- install Flask framework and gunicorn web server
- gunicorn is used by Heroku

```bash
pip install flask gunicorn
```
- gunicron is recommended web server for deploy Flask app in Heroku
    - see: https://devcenter.heroku.com/articles/python-gunicorn

### Hello World App
- follow direction here - https://flask.palletsprojects.com/en/2.1.x/quickstart/
- flask provides development server

```
cd <project folder>
export FLASK_APP=<flaskapp.py>
flask run
```

- don't need to export `FLASK_APP` env variable if the main module is named `app`
- run local web server with gunicorn
    - NOTE: do not use .py extension after `<flaskapp>`

```bash
cd <project folder>
gunicorn <flaskapp>:app
```
- see complete hello world app here: [https://github.com/rambasnet/flask-docker-mongo-heroku](https://github.com/rambasnet/flask-docker-mongo-heroku) 


### Demo applications
- `demos/flask_app_1` - a simple app with template
    - install required dependencies using the provided requirement file

```bash
cd demos/flask_app_1
pip install -r requirements.txt
export FLASK_DEBUG=True
flask run
```

- `demos/flask_app_2` - a Flask app with form
    - install required dependencies using the provided requirement file

```bash
cd demos/flask_app_2
pip install -r requirements.txt
export FLASK_DEBUG=True
flask run
```

- `demos/movieclassifier` - ML deployed app
    - install required dependencies using the provided requirement file

```bash
cd demos/movieclassifier
pip install -r requirements.txt
export FLASK_DEBUG=True
flask run
```

- `demos/movieclassifier_with_update` - ML deployed app with model update on the fly
    - install required dependencies using the provided requirement file
    
```bash
cd demos/movieclassifier_with_update
pip install -r requirements.txt
export FLASK_DEBUG=True
flask run
```

### Deploy Flask App to Heroku Platform

- Detial instruction - https://devcenter.heroku.com/articles/getting-started-with-python
- see various options here - https://flask.palletsprojects.com/en/2.1.x/deploying/
- let's use Heroku platform to deploy our Flask app
- create Heroku account
- login in to your Heroku account using browser
- download and install Heroku CLI - https://devcenter.heroku.com/articles/heroku-cli
- create an app on heroku or create it using Heroku CLI
- login in suing Heroku CLI
- add heroko to existing git repo or create a new one and add heroku
- move the demo folder (**movieclassifier_with_update**) outside existing git repository
- follow the instructions found here: https://devcenter.heroku.com/articles/git
    
- create `requirements.txt` file with Python dependencies for you project

```
cd <projectRepo>
pip list --format=freeze > requirements.txt
    
```
- create **runtime.txt** file and add python version that's supported by Heroku (similar to local version)

```
python-3.9.4
```

- create `Procfile` and add the following contents:

```
web: gunicorn hello:app
```

- `IMPORTANT` - note the required space before `gunicorn`
    - app will not launch without it as of April 12, 2021
- tell Heroku to run web server with gunicorn hello.py as the main app file

- deploy the app using heroku CLI
- must add and commit to your repository first before pushing to heroku

```bash
conda activate heroku
cd <app_folder>
git init # make sure the app is a git root repo
git branch -m master main # if necessary
git status
heroku login
heroku create "heroku-sub-dmian-app-name"
git add <file...>
git commit -m "..."
git push origin main # push to github if necessary
git push heroku main # push the contents to heroku
heroku open # open the app on a browser
```

- if successfully deployed, visit `<app-name>.herokuapp.com` or run the app from your Heroku dashboard


