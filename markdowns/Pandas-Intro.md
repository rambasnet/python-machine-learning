# Intro to Pandas
- [https://pandas.pydata.org/](https://pandas.pydata.org/)
- a fast, powerful, flexible and easy to use open source data analysis and manipulation tool built on Python

## What kind of data does pandas handle?

### pandas data table representation
![img](images/Pandas-Table.svg)
- to work with pandas package, must first import the package
- to install/update pandas you can use either conda or pip

```bash
conda install pandas
pip install pandas
```


```python
import pandas as pd
import numpy as np
```


```python
print(f'pandas version: {pd.__version__}')
print(f'numpy version: {np.__version__}')
```

## Series

- series is 1-d labeled array capable of holding any data type (integers, strings, float, Python objects, etc.)
- the axis labels collectively referred to as the **index**
- API to create Series:
```python
s = pd.Series(data, index=index)
```

- data can be:
    - NumPy's **ndarray**
    - Python dictionary
    - a scalar value (e.g. 5)
    - Python List
- index is a list of axis labels
    - index can be thought as row id or sample id
- if data is an ndarray, index must be the same length as data
    - if no index is passed, default index will be created `[0, ..., len(data)-1]`
- each column in the DataFrame is a Series


```python
s = pd.Series(np.random.randn(5))
```


```python
s
```


```python
s.index
```


```python
s1 = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
```


```python
s1
```


```python
s1.index
```


```python
# from dict
d = {"b": 1, "a": 0, "c": 2}
s2 = pd.Series(d)
```


```python
s2
```


```python
# scalar value is repeated to match the length of index
s3 = pd.Series(5.0, index=["a", "b", "c", "d", "e"])
```


```python
s3
```


```python
# creating Series from Python List
s4 = pd.Series([1, 3, 5, np.nan, 6, 8])
```


```python
s4
```

### Series is ndarray-like
- Series acts very similar to a ndarray, and is a valid argument to most NumPy functions
- slicing Series also slices the index


```python
s1
```


```python
s1[0]
```


```python
s1[3:]
```


```python
# slice using condition
s1[s1 > s1.median()]
```


```python
# slice using indices
s1[[4, 3, 1]]
```


```python
# calculate the exponential (2*n) of each element n in the ndarray
# https://numpy.org/doc/stable/reference/generated/numpy.exp.html?highlight=exp#numpy.exp
np.exp2(s3)
```


```python
s3.dtype
```

### extract data array from Series
- extract just data as array without index


```python
s3.array
```

### convert series to ndarray


```python
ndarr = s3.to_numpy()
```


```python
ndarr
```


```python
type(ndarr)
```


```python
ndarr.size
```


```python
ndarr.shape
```

### Series is dict-like
- use index as the key to get the corresponding value


```python
s1['a']
```


```python
s3['e']
```


```python
s3['e'] = 15.0
```


```python
s3
```


```python
s3['g']
```


```python
# use get with default value if key is missing
s3.get('g', np.nan)
```

### Vectorized operations and label alignment with Series
- very similar to NumPy ndarray


```python
s3
```


```python
s3+s3
```


```python
s3-s3
```


```python
s3*2
```


```python
s3/5
```


```python
# Series automatically aligns the data based on label
# if the label is not found in one Series or the other, the result will be marked as missing NaN
s3[1:] + s3[:-1]
```

### Name attribute
- Series can also have a **name** atribute


```python
s4 = pd.Series(np.random.randn(5), name="Some Name")
```


```python
s4.name
```


```python
s4
```


```python
# Series.rename creates a new Series with new name
s5 = s4.rename('New Name')
```


```python
s5
```

## DataFrame
- data table in pandas is called DataFrame
- DataFrame is the primary data structure of pandas
- Python dict can be used create DataFrame where keys will be used as column headers and the list of values as columns of the DataFrame
- each column of DataFrame is called `Series`


```python
aDict = {
    "Name": [
            "Braund, Mr. Owen Harris",
            "Allen, Mr. William Henry",
            "Bonnell, Miss. Elizabeth"
    ],
    "Age": [22, 35, 58],
    "Sex": ["male", "male", "female"]
}
```


```python
df = pd.DataFrame(aDict)
```


```python
df
```


```python
df2 = pd.DataFrame(
{
    "A": 1.0,
    "B": pd.Timestamp("20130102"),
    "C": pd.Series(1, index=list(range(4)), dtype="float32"),
    "D": np.array([3] * 4, dtype="int32"),
    "E": pd.Categorical(["test", "train", "test", "train"]),
    "F": "foo",
})
```


```python
df2
```

### spreadsheet data
- the above df can be represented in a spreadsheet software
![SpreadSheet](./images/01_table_spreadsheet.png)


```python
df
```


```python
# just work with the data in column - Age
# use dictionary syntax
df["Age"]
```


```python
# access series/column as attribute
df.Age
```

## DataFrame Complete Reference
- complete reference: https://pandas.pydata.org/pandas-docs/stable/reference/frame.html

## DataFrame utility methods and attributes to review data
- `df.columns` - return the column labels of the DataFrame
- `df.index` - return the index (row labels/ids) of the DataFrame df object
- `df.dtypes` - return Series with the data type of each column in the df object
- `df.values` - return a **NumPy** representation of the DataFrame df object
- `df.axes` - return a list representing the axes of the DataFrame, `[row labels]` and `[column labels]`
- `df.shape` - return a tuple representing the dimensionality of the DataFrame df object
- `df.size` - return an int representing the number of elements in the DataFrame df object
- `df.info()` - print a concise summary of a DataFrame df object
- `df.describe()` - generate descriptive statistics
- `df.head(n)` - display the first n rows in the DataFarme df object; default n=5
- `df.tail(n)` - display the last n rows in the DataFrame df object; default n=5


```python
df.columns
```


```python
df.index
```


```python
df.values
```


```python
df.dtypes
```


```python
df.axes
```


```python
df.shape
```


```python
df.size
```


```python
# generate descriptive statistics
df.describe()
```


```python
# print first 2 rows
df.head(2)
```


```python
# get individual stats for each Searies
df['Age'].max()
```


```python
# print last 2 rows
df.tail(2)
```

## Read and write tabular data

![](./images/02_io_readwrite1.svg)

- use pandas `.read_*(fileName)` to read data from various formats
- Pandas raw data: [https://github.com/pandas-dev/pandas/tree/master/doc/data](https://github.com/pandas-dev/pandas/tree/master/doc/data)
- read_csv - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html


```python
# read CSV file directly from the Internet
iris_df = pd.read_csv('https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/iris.data')
```


```python
iris_df.head()
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
      <th>SepalLength</th>
      <th>SepalWidth</th>
      <th>PetalLength</th>
      <th>PetalWidth</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>




```python
iris_df.tail()
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
      <th>SepalLength</th>
      <th>SepalWidth</th>
      <th>PetalLength</th>
      <th>PetalWidth</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>145</th>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>146</th>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>1.9</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>147</th>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>148</th>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>149</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
      <td>Iris-virginica</td>
    </tr>
  </tbody>
</table>
</div>




```python
# technical summary of DataFrame
iris_df.info()
```


```python
# statistical summary of iris dataset
iris_df.describe()
```

### Titanic Dataset

- https://github.com/pandas-dev/pandas/blob/master/doc/data/titanic.csv
- https://www.openml.org/d/40945
- manually download titanic.csv or use wget/curl to download the file

```bash
wget https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/titanic.csv
curl https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/titanic.csv -o data/titanic.csv
```

#### Column name description

```
PassengerId: Id of every passenger.

Survived: This feature have value 0 and 1. 0 for not survived and 1 for survived.

Pclass: Passenger class: 3 classes: Class 1, Class 2 and Class 3.

Name: Name of passenger.

Sex: Gender of passenger.

Age: Age of passenger.

SibSp: Indication that passenger have siblings and spouse.

Parch: Whether a passenger is alone or have family.

Ticket: Ticket number of passenger.

Fare: Indicating the fare.

Cabin: The cabin of passenger.

Embarked: The embarked category.
```


```bash
%%bash
mkdir -p data
curl https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/titanic.csv -o data/titanic.csv
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100 60080  100 60080    0     0   133k      0 --:--:-- --:--:-- --:--:--     0  0 --:--:-- --:--:-- --:--:--     0:-- --:--:-- --:--:--  132k



```python
# let's read titanic.csv file as DataFrame
titanicDf = pd.read_csv('data/titanic.csv')
```


```python
titanicDf
# notice the dataset already provides PassengerId as index column
# read_csv automatically adds the index column or row id
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>887</td>
      <td>0</td>
      <td>2</td>
      <td>Montvila, Rev. Juozas</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>211536</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>887</th>
      <td>888</td>
      <td>1</td>
      <td>1</td>
      <td>Graham, Miss Margaret Edith</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>112053</td>
      <td>30.0000</td>
      <td>B42</td>
      <td>S</td>
    </tr>
    <tr>
      <th>888</th>
      <td>889</td>
      <td>0</td>
      <td>3</td>
      <td>Johnston, Miss Catherine Helen "Carrie"</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>W./C. 6607</td>
      <td>23.4500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>889</th>
      <td>890</td>
      <td>1</td>
      <td>1</td>
      <td>Behr, Mr. Karl Howell</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>111369</td>
      <td>30.0000</td>
      <td>C148</td>
      <td>C</td>
    </tr>
    <tr>
      <th>890</th>
      <td>891</td>
      <td>0</td>
      <td>3</td>
      <td>Dooley, Mr. Patrick</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>370376</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 12 columns</p>
</div>




```python
# let's read the csv with PassengerId as index column (row_id)
titanicDf = pd.read_csv('data/titanic.csv', index_col="PassengerId")
```


```python
# print first 8 rows
titanicDf.head(8)
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
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>3</td>
      <td>Moran, Mr. James</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>330877</td>
      <td>8.4583</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>1</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>male</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>17463</td>
      <td>51.8625</td>
      <td>E46</td>
      <td>S</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>3</td>
      <td>Palsson, Master Gosta Leonard</td>
      <td>male</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>349909</td>
      <td>21.0750</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanicDf.shape
```


```python
titanicDf.dtypes
```

## Sort table rows
- based on some column name
- https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html

```python
DataFrame.sort_values(by='columnName', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last', ignore_index=False, key=None)
```

- returns the sorted DataFrame (NOT an inplace sort by default)


```python
sortedTitanicDf = titanicDf.sort_values(by='Age')
```


```python
sortedTitanicDf.head()
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
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>804</th>
      <td>1</td>
      <td>3</td>
      <td>Thomas, Master Assad Alexander</td>
      <td>male</td>
      <td>0.42</td>
      <td>0</td>
      <td>1</td>
      <td>2625</td>
      <td>8.5167</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>756</th>
      <td>1</td>
      <td>2</td>
      <td>Hamalainen, Master Viljo</td>
      <td>male</td>
      <td>0.67</td>
      <td>1</td>
      <td>1</td>
      <td>250649</td>
      <td>14.5000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>645</th>
      <td>1</td>
      <td>3</td>
      <td>Baclini, Miss Eugenie</td>
      <td>female</td>
      <td>0.75</td>
      <td>2</td>
      <td>1</td>
      <td>2666</td>
      <td>19.2583</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>470</th>
      <td>1</td>
      <td>3</td>
      <td>Baclini, Miss Helene Barbara</td>
      <td>female</td>
      <td>0.75</td>
      <td>2</td>
      <td>1</td>
      <td>2666</td>
      <td>19.2583</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>79</th>
      <td>1</td>
      <td>2</td>
      <td>Caldwell, Master Alden Gates</td>
      <td>male</td>
      <td>0.83</td>
      <td>0</td>
      <td>2</td>
      <td>248738</td>
      <td>29.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
# sorting by default returns sorted DF without sorting original DF in place
titanicDf.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
# sort using multiple columns and in descending order
titanicDf.sort_values(by=['Pclass', 'Age'], ascending=False).head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>851</th>
      <td>852</td>
      <td>0</td>
      <td>3</td>
      <td>Svensson, Mr. Johan</td>
      <td>male</td>
      <td>74.0</td>
      <td>0</td>
      <td>0</td>
      <td>347060</td>
      <td>7.7750</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>116</th>
      <td>117</td>
      <td>0</td>
      <td>3</td>
      <td>Connors, Mr. Patrick</td>
      <td>male</td>
      <td>70.5</td>
      <td>0</td>
      <td>0</td>
      <td>370369</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>280</th>
      <td>281</td>
      <td>0</td>
      <td>3</td>
      <td>Duane, Mr. Frank</td>
      <td>male</td>
      <td>65.0</td>
      <td>0</td>
      <td>0</td>
      <td>336439</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>483</th>
      <td>484</td>
      <td>1</td>
      <td>3</td>
      <td>Turkula, Mrs. (Hedwig)</td>
      <td>female</td>
      <td>63.0</td>
      <td>0</td>
      <td>0</td>
      <td>4134</td>
      <td>9.5875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>326</th>
      <td>327</td>
      <td>0</td>
      <td>3</td>
      <td>Nysveen, Mr. Johan Hansen</td>
      <td>male</td>
      <td>61.0</td>
      <td>0</td>
      <td>0</td>
      <td>345364</td>
      <td>6.2375</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



### Write the DataFrame as Excel file

- install openpyxl library from Terminal; doesn't seem to work from notebook
- use pip or conda to install openpyxl
- conda is installed on Codespaces

```bash
conda activate ml
conda install -y openpyxl
```

```bash
pip install openpyxl
```


```python
! conda install -y openpyxl
```


```python
sortedTitanicDf.to_excel('data/titanic_sorted_age.xlsx', sheet_name='passengers')
```


```python
# technical summary of DataFrame
sortedTitanicDf.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 891 entries, 803 to 888
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Name         891 non-null    object 
     4   Sex          891 non-null    object 
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64  
     7   Parch        891 non-null    int64  
     8   Ticket       891 non-null    object 
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object 
     11  Embarked     889 non-null    object 
    dtypes: float64(2), int64(5), object(5)
    memory usage: 90.5+ KB


## Select a subset of a DataFrame

### Select specific columns
![](./images/03_subset_columns.svg)


```python
# copy just the Age column or Series
ages = titanicDf["Age"]
```


```python
type(ages)
```




    pandas.core.series.Series




```python
ages.shape
```




    (891,)




```python
# get age and sex columns
age_sex = titanicDf[['Age', 'Sex']]
```


```python
type(age_sex)
```




    pandas.core.frame.DataFrame




```python
age_sex.head()
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
      <th>Age</th>
      <th>Sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22.0</td>
      <td>male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38.0</td>
      <td>female</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26.0</td>
      <td>female</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35.0</td>
      <td>female</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>male</td>
    </tr>
  </tbody>
</table>
</div>




```python
age_sex.shape
```




    (891, 2)




```python
# boolen mask of passengers older than 35; returns True or False based on condition
titanicDf['Age'] > 35
```




    0      False
    1       True
    2      False
    3      False
    4      False
           ...  
    886    False
    887    False
    888    False
    889    False
    890    False
    Name: Age, Length: 891, dtype: bool




```python
# DF of passengers older than 35
# passengers older than 35; returns True or False based on condition
titanicDf[titanicDf['Age'] > 35]
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>male</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>17463</td>
      <td>51.8625</td>
      <td>E46</td>
      <td>S</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>1</td>
      <td>1</td>
      <td>Bonnell, Miss Elizabeth</td>
      <td>female</td>
      <td>58.0</td>
      <td>0</td>
      <td>0</td>
      <td>113783</td>
      <td>26.5500</td>
      <td>C103</td>
      <td>S</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>0</td>
      <td>3</td>
      <td>Andersson, Mr. Anders Johan</td>
      <td>male</td>
      <td>39.0</td>
      <td>1</td>
      <td>5</td>
      <td>347082</td>
      <td>31.2750</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>1</td>
      <td>2</td>
      <td>Hewlett, Mrs. (Mary D Kingcome)</td>
      <td>female</td>
      <td>55.0</td>
      <td>0</td>
      <td>0</td>
      <td>248706</td>
      <td>16.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>865</th>
      <td>866</td>
      <td>1</td>
      <td>2</td>
      <td>Bystrom, Mrs. (Karolina)</td>
      <td>female</td>
      <td>42.0</td>
      <td>0</td>
      <td>0</td>
      <td>236852</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>871</th>
      <td>872</td>
      <td>1</td>
      <td>1</td>
      <td>Beckwith, Mrs. Richard Leonard (Sallie Monypeny)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>1</td>
      <td>11751</td>
      <td>52.5542</td>
      <td>D35</td>
      <td>S</td>
    </tr>
    <tr>
      <th>873</th>
      <td>874</td>
      <td>0</td>
      <td>3</td>
      <td>Vander Cruyssen, Mr. Victor</td>
      <td>male</td>
      <td>47.0</td>
      <td>0</td>
      <td>0</td>
      <td>345765</td>
      <td>9.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>879</th>
      <td>880</td>
      <td>1</td>
      <td>1</td>
      <td>Potter, Mrs. Thomas Jr (Lily Alexenia Wilson)</td>
      <td>female</td>
      <td>56.0</td>
      <td>0</td>
      <td>1</td>
      <td>11767</td>
      <td>83.1583</td>
      <td>C50</td>
      <td>C</td>
    </tr>
    <tr>
      <th>885</th>
      <td>886</td>
      <td>0</td>
      <td>3</td>
      <td>Rice, Mrs. William (Margaret Norton)</td>
      <td>female</td>
      <td>39.0</td>
      <td>0</td>
      <td>5</td>
      <td>382652</td>
      <td>29.1250</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
<p>217 rows × 12 columns</p>
</div>



## Select specific rows and columns
- create new DataFrame using the filtered rows and columns

- two ways:

### df.iloc selections
- use row ids and column ids
```python
df.iloc[[row selection], [column selection]]
```
- row selection can be:
    - single row index values: `[100]`
    - integer list of row indices: `[0, 2, 10]`
    - slice of row indices: `[2:10]`
        
- column selection can be:
    - single column selection: `[10]`
    - integer list of col indices: `[0, 3, 5]`
    - slice of column indices: `[3:10]`
    

### df.loc selection
- use row labels column labels

```python
df.loc[[row selection], [column selection]]
```
- row selection:
    - single row label/index: `["john"]`
    - list of row labels: `["john", "sarah"]`
    - condition: `[data['age'] >= 35]`
- column selection:
    - single column name name: `['Age']`
    - list of column names: `['Name, 'Age', 'Sex']`
    - slice of column names: `['Name':'Age']`
    
    
### Select specific rows and all the columns
![](images/03_subset_rows.svg)


```python
# Create new DataFrame based on the criteria
# similar to using where clause in SQL
passengers = titanicDf[titanicDf['Age']>35]
```


```python
passengers.head()
```


```python
passengers.describe()
```


```python
# slect all passengers who survived - rows with Survived column = 1
survived = titanicDf[titanicDf['Survived'] == 1]
```


```python
survived
```


```python
# another example of selection
class_23 = titanicDf[titanicDf['Pclass'].isin([2, 3])]
```


```python
class_23.head()
```


```python
# select data where age is known
age_not_na = titanicDf[titanicDf['Age'].notna()]
```


```python
age_not_na.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
age_not_na.shape
```




    (714, 12)




```python
# select rows 10-25 and columns 3-5
titanicDf.iloc[9:25, 2:5]
```


```python
# select rows based on row_ids or PassengerId
titanicDf.loc[[1, 3]]
```


```python
# slect rows based on row_ids and columns based on column ids
titanicDf.loc[[1, 3], ['Age', 'Name']]
```


```python
adult_names = titanicDf.loc[titanicDf['Age']>=18, ['Name']]
```


```python
adult_names.head()
```


```python
# select passengers names older than 35 years
# NOTE: loc selects based on row or column names not id
adult_age_names = titanicDf.loc[titanicDf['Age'] > 35, ['Age', 'Name']]
```


```python
adult_age_names.head()
```


```python
# TODO: select Age and Name of all the minor passengers with age less than 18
```

## Updating selected fields with iloc and loc
- update first 3 rows' Name column to "anonymous"
- `iloc` uses 0-based indices for rows and columns


```python
# Note: PassengerId is row index not part of column
titanicDf.iloc[0:3, 2] = 'anonymous'
```


```python
titanicDf.head()
```


```python
# update Name of all the children's age < 13 to anonymous
titanicDf.loc[titanicDf['Age'] < 13, ['Name']] = 'anonymous'
```


```python
# let's select and print just the Name column
titanicDf.loc[titanicDf['Age'] < 13, ['Name']]
```

## Creating new columns derived from existing columns

![](./images/05_newcolumn_1.svg)
- similar to adding just another Series with the column name as the key in DataFrame dictionary
- the calculation of the values is done **element_wise**
- remember, broadcast method?
    - you don't need to use loop to iterate each of the rows
- syntax:

```python
df['new_column_name'] = pd.Series()
```

## Open Air Quality Data
- OpenAQ Data - [https://openaq.org/#/](https://openaq.org/#/)
- register and use the API key and python library - https://python.openaq.org/tutorial/getting-started/ 
- Need to register to get an API key - https://docs.openaq.org/using-the-api/api-key 
- API documentation - https://api.openaq.org/docs

## Spent few hours trying to figure out how to get the pollution data from openaq.org, but couldn't get it to work!

## Combine data from multiple tables
- `pd.concat()` performs concatenatoins operations of multiple tables along one of the axis (row-wise or column-wise)
- typically row-wise concatenation is a common operation
- `concat` is general function provided in pandas module
    - https://pandas.pydata.org/pandas-docs/stable/reference/general_functions.html
- https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html?highlight=pandas%20concat#pandas.concat

![](./images/08_concat_row1.svg)


```python
# make a deep copy of dataframe/table
iris_df1 = iris_df.copy(deep=True)
```


```python
iris_df.shape
```




    (150, 5)




```python
iris_df1.shape
```




    (150, 5)




```python
# let's concatenate the two into a single table
combinedDF = pd.concat([iris_df, iris_df1], axis=0)
```


```python
combinedDF.shape
```




    (300, 5)



## Join tables using a common identifier
- merge tables column-wise
- the figures below show a left-join

![](./images/08_merge_left.svg)

- can use `pd.merge()` general function provided in pandas module
    - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.merge.html
- DataFrame class also provides merge method
- merge method provides how parameter to do various types of joins
    - 'left', 'right', 'outer', 'inner', 'cross', default 'inner'
    - `left`: use only keys from left frame, similar to a SQL left outer join; preserve key order.
    - `right`: use only keys from right frame, similar to a SQL right outer join; preserve key order.
    - `outer`: use union of keys from both frames, similar to a SQL full outer join; sort keys lexicographically.
    - `inner`: use intersection of keys from both frames, similar to a SQL inner join; preserve the order of the left keys.
    - `cross`: creates the cartesian product from both frames, preserves the order of the left keys.
- `pandas.concat([df1, df2], axis=0)` is equivalent to `union` in SQL

![](./images/sqlJoins_7.webp)


```python
# create a DF with key column lkey
df1 = pd.DataFrame({'lkey': ['A', 'B'],
                    'value': [1, 2]})
```


```python
df1
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
      <th>lkey</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# create a DF with key colum rkey
df2 = pd.DataFrame({'rkey': ['A', 'C', 'D'],
                    'value': [1, 5, 6]})
```


```python
df2
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
      <th>rkey</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>D</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
# cross join
df1.merge(df2, how='cross')
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
      <th>lkey</th>
      <th>value_x</th>
      <th>rkey</th>
      <th>value_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>1</td>
      <td>A</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A</td>
      <td>1</td>
      <td>C</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A</td>
      <td>1</td>
      <td>D</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B</td>
      <td>2</td>
      <td>A</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B</td>
      <td>2</td>
      <td>C</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>B</td>
      <td>2</td>
      <td>D</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
# inersection or inner join
df1.merge(df2, how='inner')
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
      <th>lkey</th>
      <th>value</th>
      <th>rkey</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>1</td>
      <td>A</td>
    </tr>
  </tbody>
</table>
</div>




```python
# left join
df1.merge(df2, how='left')
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
      <th>lkey</th>
      <th>value</th>
      <th>rkey</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>1</td>
      <td>A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# right join
df1.merge(df2, how='right')
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
      <th>lkey</th>
      <th>value</th>
      <th>rkey</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>1</td>
      <td>A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>5</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>6</td>
      <td>D</td>
    </tr>
  </tbody>
</table>
</div>




```python
# outer join
df1.merge(df2, how='outer')
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
      <th>lkey</th>
      <th>value</th>
      <th>rkey</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>1</td>
      <td>A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>5</td>
      <td>C</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>6</td>
      <td>D</td>
    </tr>
  </tbody>
</table>
</div>




```python
# union
pd.concat([df1, df2], axis=0)
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
      <th>lkey</th>
      <th>value</th>
      <th>rkey</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>1</td>
      <td>A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>5</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>6</td>
      <td>D</td>
    </tr>
  </tbody>
</table>
</div>




```python
no2_url = 'https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/air_quality_no2_long.csv'
pm2_url = 'https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/air_quality_pm25_long.csv'
air_quality_stations_url = 'https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/air_quality_stations.csv'
air_qual_parameters_url = 'https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/air_quality_parameters.csv'
```


```python
air_quality_no2 = pd.read_csv(no2_url)
```


```python
air_quality_no2.head()
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
      <th>city</th>
      <th>country</th>
      <th>date.utc</th>
      <th>location</th>
      <th>parameter</th>
      <th>value</th>
      <th>unit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Paris</td>
      <td>FR</td>
      <td>2019-06-21 00:00:00+00:00</td>
      <td>FR04014</td>
      <td>no2</td>
      <td>20.0</td>
      <td>µg/m³</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Paris</td>
      <td>FR</td>
      <td>2019-06-20 23:00:00+00:00</td>
      <td>FR04014</td>
      <td>no2</td>
      <td>21.8</td>
      <td>µg/m³</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Paris</td>
      <td>FR</td>
      <td>2019-06-20 22:00:00+00:00</td>
      <td>FR04014</td>
      <td>no2</td>
      <td>26.5</td>
      <td>µg/m³</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Paris</td>
      <td>FR</td>
      <td>2019-06-20 21:00:00+00:00</td>
      <td>FR04014</td>
      <td>no2</td>
      <td>24.9</td>
      <td>µg/m³</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Paris</td>
      <td>FR</td>
      <td>2019-06-20 20:00:00+00:00</td>
      <td>FR04014</td>
      <td>no2</td>
      <td>21.4</td>
      <td>µg/m³</td>
    </tr>
  </tbody>
</table>
</div>




```python
air_quality_parameters = pd.read_csv(air_qual_parameters_url)
```


```python
air_quality_parameters.head()
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
      <th>id</th>
      <th>description</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bc</td>
      <td>Black Carbon</td>
      <td>BC</td>
    </tr>
    <tr>
      <th>1</th>
      <td>co</td>
      <td>Carbon Monoxide</td>
      <td>CO</td>
    </tr>
    <tr>
      <th>2</th>
      <td>no2</td>
      <td>Nitrogen Dioxide</td>
      <td>NO2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>o3</td>
      <td>Ozone</td>
      <td>O3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pm10</td>
      <td>Particulate matter less than 10 micrometers in...</td>
      <td>PM10</td>
    </tr>
  </tbody>
</table>
</div>




```python
# column parameter in air_quality_no2 table and id in air_quality_parameters are common
air_quality = pd.merge(air_quality_no2, air_quality_parameters, how='left', left_on='parameter', right_on='id')
```


```python
air_quality.head(10)
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
      <th>city</th>
      <th>country</th>
      <th>date.utc</th>
      <th>location</th>
      <th>parameter</th>
      <th>value</th>
      <th>unit</th>
      <th>id</th>
      <th>description</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Paris</td>
      <td>FR</td>
      <td>2019-06-21 00:00:00+00:00</td>
      <td>FR04014</td>
      <td>no2</td>
      <td>20.0</td>
      <td>µg/m³</td>
      <td>no2</td>
      <td>Nitrogen Dioxide</td>
      <td>NO2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Paris</td>
      <td>FR</td>
      <td>2019-06-20 23:00:00+00:00</td>
      <td>FR04014</td>
      <td>no2</td>
      <td>21.8</td>
      <td>µg/m³</td>
      <td>no2</td>
      <td>Nitrogen Dioxide</td>
      <td>NO2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Paris</td>
      <td>FR</td>
      <td>2019-06-20 22:00:00+00:00</td>
      <td>FR04014</td>
      <td>no2</td>
      <td>26.5</td>
      <td>µg/m³</td>
      <td>no2</td>
      <td>Nitrogen Dioxide</td>
      <td>NO2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Paris</td>
      <td>FR</td>
      <td>2019-06-20 21:00:00+00:00</td>
      <td>FR04014</td>
      <td>no2</td>
      <td>24.9</td>
      <td>µg/m³</td>
      <td>no2</td>
      <td>Nitrogen Dioxide</td>
      <td>NO2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Paris</td>
      <td>FR</td>
      <td>2019-06-20 20:00:00+00:00</td>
      <td>FR04014</td>
      <td>no2</td>
      <td>21.4</td>
      <td>µg/m³</td>
      <td>no2</td>
      <td>Nitrogen Dioxide</td>
      <td>NO2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Paris</td>
      <td>FR</td>
      <td>2019-06-20 19:00:00+00:00</td>
      <td>FR04014</td>
      <td>no2</td>
      <td>25.3</td>
      <td>µg/m³</td>
      <td>no2</td>
      <td>Nitrogen Dioxide</td>
      <td>NO2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Paris</td>
      <td>FR</td>
      <td>2019-06-20 18:00:00+00:00</td>
      <td>FR04014</td>
      <td>no2</td>
      <td>23.9</td>
      <td>µg/m³</td>
      <td>no2</td>
      <td>Nitrogen Dioxide</td>
      <td>NO2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Paris</td>
      <td>FR</td>
      <td>2019-06-20 17:00:00+00:00</td>
      <td>FR04014</td>
      <td>no2</td>
      <td>23.2</td>
      <td>µg/m³</td>
      <td>no2</td>
      <td>Nitrogen Dioxide</td>
      <td>NO2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Paris</td>
      <td>FR</td>
      <td>2019-06-20 16:00:00+00:00</td>
      <td>FR04014</td>
      <td>no2</td>
      <td>19.0</td>
      <td>µg/m³</td>
      <td>no2</td>
      <td>Nitrogen Dioxide</td>
      <td>NO2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Paris</td>
      <td>FR</td>
      <td>2019-06-20 15:00:00+00:00</td>
      <td>FR04014</td>
      <td>no2</td>
      <td>19.3</td>
      <td>µg/m³</td>
      <td>no2</td>
      <td>Nitrogen Dioxide</td>
      <td>NO2</td>
    </tr>
  </tbody>
</table>
</div>




```python
air_quality.tail(10)
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
      <th>city</th>
      <th>country</th>
      <th>date.utc</th>
      <th>location</th>
      <th>parameter</th>
      <th>value</th>
      <th>unit</th>
      <th>id</th>
      <th>description</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2058</th>
      <td>London</td>
      <td>GB</td>
      <td>2019-05-07 11:00:00+00:00</td>
      <td>London Westminster</td>
      <td>no2</td>
      <td>21.0</td>
      <td>µg/m³</td>
      <td>no2</td>
      <td>Nitrogen Dioxide</td>
      <td>NO2</td>
    </tr>
    <tr>
      <th>2059</th>
      <td>London</td>
      <td>GB</td>
      <td>2019-05-07 10:00:00+00:00</td>
      <td>London Westminster</td>
      <td>no2</td>
      <td>21.0</td>
      <td>µg/m³</td>
      <td>no2</td>
      <td>Nitrogen Dioxide</td>
      <td>NO2</td>
    </tr>
    <tr>
      <th>2060</th>
      <td>London</td>
      <td>GB</td>
      <td>2019-05-07 09:00:00+00:00</td>
      <td>London Westminster</td>
      <td>no2</td>
      <td>28.0</td>
      <td>µg/m³</td>
      <td>no2</td>
      <td>Nitrogen Dioxide</td>
      <td>NO2</td>
    </tr>
    <tr>
      <th>2061</th>
      <td>London</td>
      <td>GB</td>
      <td>2019-05-07 08:00:00+00:00</td>
      <td>London Westminster</td>
      <td>no2</td>
      <td>32.0</td>
      <td>µg/m³</td>
      <td>no2</td>
      <td>Nitrogen Dioxide</td>
      <td>NO2</td>
    </tr>
    <tr>
      <th>2062</th>
      <td>London</td>
      <td>GB</td>
      <td>2019-05-07 07:00:00+00:00</td>
      <td>London Westminster</td>
      <td>no2</td>
      <td>32.0</td>
      <td>µg/m³</td>
      <td>no2</td>
      <td>Nitrogen Dioxide</td>
      <td>NO2</td>
    </tr>
    <tr>
      <th>2063</th>
      <td>London</td>
      <td>GB</td>
      <td>2019-05-07 06:00:00+00:00</td>
      <td>London Westminster</td>
      <td>no2</td>
      <td>26.0</td>
      <td>µg/m³</td>
      <td>no2</td>
      <td>Nitrogen Dioxide</td>
      <td>NO2</td>
    </tr>
    <tr>
      <th>2064</th>
      <td>London</td>
      <td>GB</td>
      <td>2019-05-07 04:00:00+00:00</td>
      <td>London Westminster</td>
      <td>no2</td>
      <td>16.0</td>
      <td>µg/m³</td>
      <td>no2</td>
      <td>Nitrogen Dioxide</td>
      <td>NO2</td>
    </tr>
    <tr>
      <th>2065</th>
      <td>London</td>
      <td>GB</td>
      <td>2019-05-07 03:00:00+00:00</td>
      <td>London Westminster</td>
      <td>no2</td>
      <td>19.0</td>
      <td>µg/m³</td>
      <td>no2</td>
      <td>Nitrogen Dioxide</td>
      <td>NO2</td>
    </tr>
    <tr>
      <th>2066</th>
      <td>London</td>
      <td>GB</td>
      <td>2019-05-07 02:00:00+00:00</td>
      <td>London Westminster</td>
      <td>no2</td>
      <td>19.0</td>
      <td>µg/m³</td>
      <td>no2</td>
      <td>Nitrogen Dioxide</td>
      <td>NO2</td>
    </tr>
    <tr>
      <th>2067</th>
      <td>London</td>
      <td>GB</td>
      <td>2019-05-07 01:00:00+00:00</td>
      <td>London Westminster</td>
      <td>no2</td>
      <td>23.0</td>
      <td>µg/m³</td>
      <td>no2</td>
      <td>Nitrogen Dioxide</td>
      <td>NO2</td>
    </tr>
  </tbody>
</table>
</div>



## Working with textual data
- can apply all the Python string methods on text data
- let's work on Titanic dataset


```python
import pandas as pd
```


```python
titanic = pd.read_csv('data/titanic.csv', index_col="PassengerId")
```


```python
titanic.head()
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
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
# convert Names to lowercase
titanic["Name"] = titanic["Name"].str.lower()
```


```python
titanic.head()
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
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>3</td>
      <td>braund, mr. owen harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>cumings, mrs. john bradley (florence briggs th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>3</td>
      <td>heikkinen, miss laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>futrelle, mrs. jacques heath (lily may peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>3</td>
      <td>allen, mr. william henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create a new column "Surname" that contains the last name by extracting the part before the comma in Name
titanic["Name"].str.split(",")
```




    PassengerId
    1                             [braund,  mr. owen harris]
    2      [cumings,  mrs. john bradley (florence briggs ...
    3                               [heikkinen,  miss laina]
    4        [futrelle,  mrs. jacques heath (lily may peel)]
    5                            [allen,  mr. william henry]
                                 ...                        
    887                             [montvila,  rev. juozas]
    888                       [graham,  miss margaret edith]
    889           [johnston,  miss catherine helen "carrie"]
    890                             [behr,  mr. karl howell]
    891                               [dooley,  mr. patrick]
    Name: Name, Length: 891, dtype: object




```python
titanic["Surname"] = titanic["Name"].str.split(",").str.get(0)
```


```python
titanic.head()
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
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Surname</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>3</td>
      <td>braund, mr. owen harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>braund</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>cumings, mrs. john bradley (florence briggs th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>cumings</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>3</td>
      <td>heikkinen, miss laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>heikkinen</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>futrelle, mrs. jacques heath (lily may peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>futrelle</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>3</td>
      <td>allen, mr. william henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>allen</td>
    </tr>
  </tbody>
</table>
</div>




```python
# extract the passengers info with the Name that contains "henry" on board of the Titanic
titanic[titanic["Name"].str.contains("henry")]
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
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Surname</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>3</td>
      <td>allen, mr. william henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>allen</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
      <td>3</td>
      <td>saundercock, mr. william henry</td>
      <td>male</td>
      <td>20.0</td>
      <td>0</td>
      <td>0</td>
      <td>A/5. 2151</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>saundercock</td>
    </tr>
    <tr>
      <th>53</th>
      <td>1</td>
      <td>1</td>
      <td>harper, mrs. henry sleeper (myna haxtun)</td>
      <td>female</td>
      <td>49.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17572</td>
      <td>76.7292</td>
      <td>D33</td>
      <td>C</td>
      <td>harper</td>
    </tr>
    <tr>
      <th>63</th>
      <td>0</td>
      <td>1</td>
      <td>harris, mr. henry birkhardt</td>
      <td>male</td>
      <td>45.0</td>
      <td>1</td>
      <td>0</td>
      <td>36973</td>
      <td>83.4750</td>
      <td>C83</td>
      <td>S</td>
      <td>harris</td>
    </tr>
    <tr>
      <th>160</th>
      <td>0</td>
      <td>3</td>
      <td>sage, master thomas henry</td>
      <td>male</td>
      <td>NaN</td>
      <td>8</td>
      <td>2</td>
      <td>CA. 2343</td>
      <td>69.5500</td>
      <td>NaN</td>
      <td>S</td>
      <td>sage</td>
    </tr>
    <tr>
      <th>177</th>
      <td>0</td>
      <td>3</td>
      <td>lefebre, master henry forbes</td>
      <td>male</td>
      <td>NaN</td>
      <td>3</td>
      <td>1</td>
      <td>4133</td>
      <td>25.4667</td>
      <td>NaN</td>
      <td>S</td>
      <td>lefebre</td>
    </tr>
    <tr>
      <th>210</th>
      <td>1</td>
      <td>1</td>
      <td>blank, mr. henry</td>
      <td>male</td>
      <td>40.0</td>
      <td>0</td>
      <td>0</td>
      <td>112277</td>
      <td>31.0000</td>
      <td>A31</td>
      <td>C</td>
      <td>blank</td>
    </tr>
    <tr>
      <th>213</th>
      <td>0</td>
      <td>3</td>
      <td>perkin, mr. john henry</td>
      <td>male</td>
      <td>22.0</td>
      <td>0</td>
      <td>0</td>
      <td>A/5 21174</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>perkin</td>
    </tr>
    <tr>
      <th>223</th>
      <td>0</td>
      <td>3</td>
      <td>green, mr. george henry</td>
      <td>male</td>
      <td>51.0</td>
      <td>0</td>
      <td>0</td>
      <td>21440</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>green</td>
    </tr>
    <tr>
      <th>228</th>
      <td>0</td>
      <td>3</td>
      <td>lovell, mr. john hall ("henry")</td>
      <td>male</td>
      <td>20.5</td>
      <td>0</td>
      <td>0</td>
      <td>A/5 21173</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>lovell</td>
    </tr>
    <tr>
      <th>231</th>
      <td>1</td>
      <td>1</td>
      <td>harris, mrs. henry birkhardt (irene wallach)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>36973</td>
      <td>83.4750</td>
      <td>C83</td>
      <td>S</td>
      <td>harris</td>
    </tr>
    <tr>
      <th>240</th>
      <td>0</td>
      <td>2</td>
      <td>hunt, mr. george henry</td>
      <td>male</td>
      <td>33.0</td>
      <td>0</td>
      <td>0</td>
      <td>SCO/W 1585</td>
      <td>12.2750</td>
      <td>NaN</td>
      <td>S</td>
      <td>hunt</td>
    </tr>
    <tr>
      <th>265</th>
      <td>0</td>
      <td>3</td>
      <td>henry, miss delia</td>
      <td>female</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>382649</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
      <td>henry</td>
    </tr>
    <tr>
      <th>272</th>
      <td>1</td>
      <td>3</td>
      <td>tornquist, mr. william henry</td>
      <td>male</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0000</td>
      <td>NaN</td>
      <td>S</td>
      <td>tornquist</td>
    </tr>
    <tr>
      <th>335</th>
      <td>1</td>
      <td>1</td>
      <td>frauenthal, mrs. henry william (clara heinshei...</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17611</td>
      <td>133.6500</td>
      <td>NaN</td>
      <td>S</td>
      <td>frauenthal</td>
    </tr>
    <tr>
      <th>348</th>
      <td>1</td>
      <td>3</td>
      <td>davison, mrs. thomas henry (mary e finck)</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>386525</td>
      <td>16.1000</td>
      <td>NaN</td>
      <td>S</td>
      <td>davison</td>
    </tr>
    <tr>
      <th>386</th>
      <td>0</td>
      <td>2</td>
      <td>davies, mr. charles henry</td>
      <td>male</td>
      <td>18.0</td>
      <td>0</td>
      <td>0</td>
      <td>S.O.C. 14879</td>
      <td>73.5000</td>
      <td>NaN</td>
      <td>S</td>
      <td>davies</td>
    </tr>
    <tr>
      <th>412</th>
      <td>0</td>
      <td>3</td>
      <td>hart, mr. henry</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>394140</td>
      <td>6.8583</td>
      <td>NaN</td>
      <td>Q</td>
      <td>hart</td>
    </tr>
    <tr>
      <th>477</th>
      <td>0</td>
      <td>2</td>
      <td>renouf, mr. peter henry</td>
      <td>male</td>
      <td>34.0</td>
      <td>1</td>
      <td>0</td>
      <td>31027</td>
      <td>21.0000</td>
      <td>NaN</td>
      <td>S</td>
      <td>renouf</td>
    </tr>
    <tr>
      <th>483</th>
      <td>0</td>
      <td>3</td>
      <td>rouse, mr. richard henry</td>
      <td>male</td>
      <td>50.0</td>
      <td>0</td>
      <td>0</td>
      <td>A/5 3594</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>rouse</td>
    </tr>
    <tr>
      <th>509</th>
      <td>0</td>
      <td>3</td>
      <td>olsen, mr. henry margido</td>
      <td>male</td>
      <td>28.0</td>
      <td>0</td>
      <td>0</td>
      <td>C 4001</td>
      <td>22.5250</td>
      <td>NaN</td>
      <td>S</td>
      <td>olsen</td>
    </tr>
    <tr>
      <th>595</th>
      <td>0</td>
      <td>2</td>
      <td>chapman, mr. john henry</td>
      <td>male</td>
      <td>37.0</td>
      <td>1</td>
      <td>0</td>
      <td>SC/AH 29037</td>
      <td>26.0000</td>
      <td>NaN</td>
      <td>S</td>
      <td>chapman</td>
    </tr>
    <tr>
      <th>624</th>
      <td>0</td>
      <td>3</td>
      <td>hansen, mr. henry damsgaard</td>
      <td>male</td>
      <td>21.0</td>
      <td>0</td>
      <td>0</td>
      <td>350029</td>
      <td>7.8542</td>
      <td>NaN</td>
      <td>S</td>
      <td>hansen</td>
    </tr>
    <tr>
      <th>631</th>
      <td>1</td>
      <td>1</td>
      <td>barkworth, mr. algernon henry wilson</td>
      <td>male</td>
      <td>80.0</td>
      <td>0</td>
      <td>0</td>
      <td>27042</td>
      <td>30.0000</td>
      <td>A23</td>
      <td>S</td>
      <td>barkworth</td>
    </tr>
    <tr>
      <th>634</th>
      <td>0</td>
      <td>1</td>
      <td>parr, mr. william henry marsh</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>112052</td>
      <td>0.0000</td>
      <td>NaN</td>
      <td>S</td>
      <td>parr</td>
    </tr>
    <tr>
      <th>646</th>
      <td>1</td>
      <td>1</td>
      <td>harper, mr. henry sleeper</td>
      <td>male</td>
      <td>48.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17572</td>
      <td>76.7292</td>
      <td>D33</td>
      <td>C</td>
      <td>harper</td>
    </tr>
    <tr>
      <th>661</th>
      <td>1</td>
      <td>1</td>
      <td>frauenthal, dr. henry william</td>
      <td>male</td>
      <td>50.0</td>
      <td>2</td>
      <td>0</td>
      <td>PC 17611</td>
      <td>133.6500</td>
      <td>NaN</td>
      <td>S</td>
      <td>frauenthal</td>
    </tr>
    <tr>
      <th>673</th>
      <td>0</td>
      <td>2</td>
      <td>mitchell, mr. henry michael</td>
      <td>male</td>
      <td>70.0</td>
      <td>0</td>
      <td>0</td>
      <td>C.A. 24580</td>
      <td>10.5000</td>
      <td>NaN</td>
      <td>S</td>
      <td>mitchell</td>
    </tr>
    <tr>
      <th>696</th>
      <td>0</td>
      <td>2</td>
      <td>chapman, mr. charles henry</td>
      <td>male</td>
      <td>52.0</td>
      <td>0</td>
      <td>0</td>
      <td>248731</td>
      <td>13.5000</td>
      <td>NaN</td>
      <td>S</td>
      <td>chapman</td>
    </tr>
    <tr>
      <th>706</th>
      <td>0</td>
      <td>2</td>
      <td>morley, mr. henry samuel ("mr henry marshall")</td>
      <td>male</td>
      <td>39.0</td>
      <td>0</td>
      <td>0</td>
      <td>250655</td>
      <td>26.0000</td>
      <td>NaN</td>
      <td>S</td>
      <td>morley</td>
    </tr>
    <tr>
      <th>723</th>
      <td>0</td>
      <td>2</td>
      <td>gillespie, mr. william henry</td>
      <td>male</td>
      <td>34.0</td>
      <td>0</td>
      <td>0</td>
      <td>12233</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
      <td>gillespie</td>
    </tr>
    <tr>
      <th>724</th>
      <td>0</td>
      <td>2</td>
      <td>hodges, mr. henry price</td>
      <td>male</td>
      <td>50.0</td>
      <td>0</td>
      <td>0</td>
      <td>250643</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
      <td>hodges</td>
    </tr>
    <tr>
      <th>727</th>
      <td>1</td>
      <td>2</td>
      <td>renouf, mrs. peter henry (lillian jefferys)</td>
      <td>female</td>
      <td>30.0</td>
      <td>3</td>
      <td>0</td>
      <td>31027</td>
      <td>21.0000</td>
      <td>NaN</td>
      <td>S</td>
      <td>renouf</td>
    </tr>
    <tr>
      <th>885</th>
      <td>0</td>
      <td>3</td>
      <td>sutehall, mr. henry jr</td>
      <td>male</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>SOTON/OQ 392076</td>
      <td>7.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>sutehall</td>
    </tr>
  </tbody>
</table>
</div>




```python
# select Name of the passenger with the longest Name
titanic.loc[titanic["Name"].str.len().idxmax(), "Name"]
# idxmax() gets the index label for which the length is the largest
```




    'penasco y castellana, mrs. victor de satode (maria josefa perez de soto y vallejo)'




```python
# replace values of "male" by "M" and values of "female" by "F" and add it as a new column
# replace method requires a dictionary to define the mapping {from: to}
titanic["Gender"] = titanic["Sex"].replace({"male": "M", "female": "F"})
```


```python
titanic.head()
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
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Surname</th>
      <th>Gender</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>3</td>
      <td>braund, mr. owen harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>braund</td>
      <td>M</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>cumings, mrs. john bradley (florence briggs th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>cumings</td>
      <td>F</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>3</td>
      <td>heikkinen, miss laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>heikkinen</td>
      <td>F</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>futrelle, mrs. jacques heath (lily may peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>futrelle</td>
      <td>F</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>3</td>
      <td>allen, mr. william henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>allen</td>
      <td>M</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
