# Intro To NumPy
- numpy is Python library for fast array computing (as fast as C and Fortran) and used in every field of science and engineering
- offers comprehensive mathematical functions, random number generators, linear algebra routines, Fourier transforms, and more
- foundation of scientific Python and PyData ecosystems such as:
    - Pandas, SciPy, Matplotlib, scikit-learn, scikit-image and most other data science packages
- the heart of NumPy is **ndarray**, a homogenous n-dimensional array object, with methods to efficiently operate on it
- [Beginners Guide](https://numpy.org/devdocs/user/absolute_beginners.html)
- [NumPy Fundamentals](https://numpy.org/devdocs/user/basics.html)

## Installation
- can use conda or pip

```bash
conda config --env --add channels conda-forge
conda install numpy
```

```
pip install numpy
```

## import NumPy
- must import numpy library to use in Python script; typical usage is:


```python
import numpy as np
```


```python
print(np.__version__)
```

    1.23.4



```python
array = np.arange(6)
```


```python
array.shape
```




    (6,)




```python
array
```




    array([0, 1, 2, 3, 4, 5])



## Difference between a Python list and a NumPy array
- NumPy array data has same type (homogenous)
- provides enourmous speed on mathematical operation that are meant to be performed on arrays
- Python list can contain different data types within a single list (heterogenous)
    - much slower and inefficienet in operations

## NumPy array
- central data structure of the NumPy library
- grid of elments that can be indexed in various ways
- the elements are of the same type, referred to as the array **dtype**
- the **rank** of the array is the number of dimensions
- the **shape** of the array is a tuple of integers giving the size of the array along each dimension
- can initialize NumPy arrays from Python lists


```python
a = np.array([1, 2, 3, 4, 5, 6])
```


```python
b = np.array([[1, 2, 3, 4], [10, 20, 30, 40], [100, 200, 300, 400]])
```


```python
b.shape
```




    (3, 4)




```python
# accessing np array is similar to Python list using 0-based indices
print(a[0])
```

    1



```python
print(b)
```

    [[  1   2   3   4]
     [ 10  20  30  40]
     [100 200 300 400]]



```python
print(b[2][0])
```

    100


### Types of array
- **1-D** array is also called **vector**
    - no difference between row and column vectors
- **2-D** array is also called **matrix**
- **3-d** and higher dimensional arrays are also called **tensor**

### Attributes of an array
- array is usually a fixed-size container of items of the same type and size
- the number of dimensions and items in an array is defined byt its shape
- the shape is a tuple that specify the sizes of each dimension
- NumPy dimensions are called **axes**
- the *b* NumPy **ndarray** is a 2-d matrix
- the *b* array has 2 axes
- the first axis (row) has length of 3 and the second axis (column) has a length of 4


```python
b
```




    array([[  1,   2,   3,   4],
           [ 10,  20,  30,  40],
           [100, 200, 300, 400]])



## Creating basic array
- various ways; primary is by using **np.array()**


```python
a = np.array([1, 2, 3])
```


```python
a
```


```python
# create and initialize elements with 0s
a = np.zeros(4)
```


```python
a
```




    array([0., 0., 0., 0.])




```python
# create an initialize elements with 1s
a = np.ones(5)
```


```python
a
```


```python
# create an empty array with random values; make sure to fill the array with actual elements
a = np.empty(2)
```


```python
a
```




    array([2.05833592e-312, 2.33419537e-312])




```python
# use arange(start, stop, step)
np.arange(2, 9, 2)
```




    array([2, 4, 6, 8])




```python
# create an array with values that are spaced linearly in a specified interval
np.linspace(0, 10, num=5)
```




    array([ 0. ,  2.5,  5. ,  7.5, 10. ])




```python
# specify datatype; default is np.float64
np.ones(5, dtype=np.int64)
```




    array([1, 1, 1, 1, 1])



## Adding, removing, and sorting elements
- https://numpy.org/devdocs/reference/generated/numpy.sort.html#numpy.sort
- `np.sort(a, axis=-1, kind=None, order=None)` -  array a to be sorted and return the sorted ndarray
    - axis : default-1 sorts along the last axis
    - kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, default is quicksort
    - order: str or list of str where str is field name or list of field names


```python
a = np.array([3, 1, 2, 4])
```


```python
a.sort()
```


```python
a
```


```python
b = np.array([5, 6, 7, 8])
```


```python
np.concatenate((a, b))
```


```python
np.concatenate((a, b), axis=0)
```


```python
c = np.array([7, 8, 9, 10])
```


```python
np.concatenate((a, b, c))
```


```python
# concatenate 2-d array
matrix = np.concatenate(([a], [b], [c]))
```


```python
matrix
```

## know the shape and size of array
- ndarray.shape, ndarray.size, ndarray.ndim


```python
matrix.shape
```


```python
matrix.size
# product of the elements of array's shape
```


```python
matrix.ndim
# number of axes or dimensions
```

## Indexing and slicing
- NumPy arrays can be sliced the same way as Python lists


```python
data = np.array([1, 2, 3])
```


```python
data[1]
```


```python
data[1:]
```


```python
data[-1]
```


```python
# slice array with certain conditions
a = np.array([[1 , 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
```


```python
a
```


```python
# print values in the array that are less than 5 as a 1-d array
print(a[a < 5])
```


```python
# select numbers that are equal to or greater than 5; use that condition to index an array
# keeps the original dimension of the array
five_up = a >=5
```


```python
five_up
```


```python
# select elements that satisfiy two conditions using & and | operators
c = a[(a>2) & (a<11)]
```


```python
c
```

## basic operations on arrays
- `+` - add two arrays' corresponding elements
- `-` - subtract one array from another's corresponding elements
- `*` - multiply one array by another's corresponding elements
- `/` - divide one array by another's corresponding elements


```python
data = np.array([1, 2])
ones = np.ones(2, dtype=int)
```


```python
data
```


```python
ones
```


```python
data + ones
```


```python
data - ones
```


```python
data / ones
```


```python
data.sum()
```


```python
# you specifiy the axis on 2-d array
b = np.array([[1, 1], [0.5, 0.5]])
```


```python
# sum the rows
b.sum(axis=0)
```


```python
# sum the columns
b.sum(axis=1)
```


```python
b.min()
```


```python
b.max()
```


```python
b.sum()
```


```python
# find min on each column
b.min(axis=0)
```


```python
# find min on each row
b.min(axis=1)
```

## Broadcasting
- an operation between a vector and a scalar applies to all the elements in vector


```python
data = np.array([1.0, 2.0, 3.0])
```


```python
data * 1.6
```


```python
data + 1.1
```


```python
data / 2
```


```python
data - 1
```

## Matrix computation

- linear-algebra based computation and more...
- https://numpy.org/doc/stable/reference/routines.linalg.html


```python
A = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
```


```python
A
```


```python
B = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]])
```


```python
B
```


```python
A + B
```


```python
A - B
```


```python
A * B
```


```python
A / B
```


```python
C = np.dot(A, B) 
```


```python
C
```

## Transposing and reshaping a matrix


```python
data = np.arange(1, 7, 1)
```


```python
data
```


```python
# 2x3 matrix
X = data.reshape(2, 3)
```


```python
X
```


```python
# 3x2 matrix
data.reshape(3, 2)
```


```python
X.transpose()
```


```python
# flatten n-d array to 1-d array
X.flatten()
```

## mathematical formulas
- MeanSquareError = $\frac{1}{n}\sum_{i=1}^{n}(Y\_prediction_i - Y_i)^2$


```python
predictions = np.ones(3)
labels = np.arange(1, 4)
```


```python
print(predictions, labels)
```


```python
error = 1/len(predictions)*np.sum(np.square(predictions-labels))
```


```python
print(f'supervised ML error= {error}')
```


```python

```
