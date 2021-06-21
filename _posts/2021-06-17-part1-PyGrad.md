---
layout: post
title:  "Part 1: An overview of Deep Learning"
date:   2021-06-17 12:46:08 +0530
categories: PyGrad
---

## Topics Covered
- The Basics of Deep Learning.
- Building a Simple Neural Network from Scratch (forward pass only).

## Introduction
In this blog post, I'll put forth the **big picture** and layout of how Deep Learning works. I'll be taking an example of a two layered Neural Network (Multi Layer Perceptron) and use [Diabetes Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html#sklearn.datasets.load_diabetes) to demonstrate different concepts.

I've also put together a [Jupyter Notebook](https://jupyter.org/) such that it's easy for you to follow along. Please click [this link](https://github.com/tgautam03/tgautam03.github.io/blob/master/jupyter/01-iota-intuitiveDL/01_Intuitive_DL.ipynb) to check the notebook out. 

## Dataset
Let's first look at the dataset at hand. It can be easily loaded using [scikit-learn](https://scikit-learn.org) as follows:

```python
# importing the scikit-learn library
from sklearn.datasets import load_diabetes
# importing numpy
import numpy as np

# loading diabetes dataset
diabetes = load_diabetes()
X = diabetes.data # extracting input features
Y = diabetes.target # extracting targets
features = diabetes.feature_names # extracting feature names
```

To visualise, I'll be using [pandas](https://pandas.pydata.org/). Let's first load the numpy arrays into pandas dataframe and then display it:
```python
# importing pandas library
import pandas as pd

# Loading the numpy arrays into a dataframe
df = pd.DataFrame(X, columns=features)
```
```python
# Displaying dataframe
df
```
![01.png](https://raw.githubusercontent.com/tgautam03/tgautam03.github.io/master/images/01-iota-intuitiveDL/01.png)

In the figure above you can see different features that will be used to predict *"a quantitative measure of disease progression one year after baseline"* (for details click [here](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)).

## Neural Network Architecture
The *architecture* of a Neural Network defines the *order* and *type of operations* used. In a broad sense, the Neural Network in this post will have *one Input Layer* ($$x_1$$, $$x_2$$, .., $$x_{10}$$), *one hidden layer* ($$h_i$$, $$z_i$$ and $$a_i$$) and *one output layer* ($$Y$$). There are various operations like *Matrix Multiplication*, *Bias Addition* and *Activation* defined inside/in-between these layers, which *transform* the input feature `X` to predictions `Y` as shown in the figure below. 


![02.png](https://raw.githubusercontent.com/tgautam03/tgautam03.github.io/master/images/01-iota-intuitiveDL/02.png)

- *Green Arrows*: Information from all neurons (*grey circles*) are getting mixed. This is a *matrix dot product* (as the whole) in the case of a Fully Connected Neural Network.
- *Blue Arrows*: No mixing of information between neurons. This represents *Bias Addition* on each neuron. 
- *Red Arrows*: No mixing of information between neurons. This represents elementwise *Activation* operation on each neuron.

Hence, on the lowest level, we have *operations* making up the whole Neural Network and I'll now discuss each of these in detail next.

## Deep Learning
The word *Deep Learning* signifies that the Neural Network architecture is *deep*, i.e. there are multiple instances of each *operation* (specifically *Matrix Multiplication* and *Bias Addition*, as they make up a *layer*). These *operations* are applied in specific order such that they *map* input features to labels.

For the sake of simplicity, let's first look at one example (the very first) from the dataset. Here, we have 10 independent features that will be used to make predictions.
$$ 
\begin{aligned}
X^{(0)T} =
\begin{pmatrix}
x_{0,0} \\
x_{0,1} \\ 
x_{0,2} \\
x_{0,3} \\
x_{0,4} \\
x_{0,5}\\
x_{0,6}\\
x_{0,7}\\
x_{0,8}\\
x_{0,9}
\end{pmatrix}
=
\begin{pmatrix}
0.038 \\
0.050 \\ 
0.061 \\
0.021 \\
-0.044 \\
-0.034\\
-0.043\\
-0.002\\
0.019\\
-0.017
\end{pmatrix}; \textrm{Dimension=} 10 \times 1,\ T =  \textrm{Transpose}
\end{aligned}
\\
\textrm{where } X^{(0)} \textrm{means the very first example in matrix } X
$$

> Note: While training, complete dataset is used so that the Network can generalise well over multiple examples.

The various *operations* used throughout the Deep Neural Network are described as follows:
### 1) Matrix Multiplication
During this operation, the input is multiplied by a *weights* matrix whose dimension is defined in accordance with the inputs and the desired output size. 

For Example in case of the very first layer which maps input vectors $$X^(i)$$ (of length 10) to first hidden units vectors $$H_1$$ (of length 6), the *weights* matrix is defined as

$$ 
W_1 =
\begin{pmatrix}
w_{0,0} && w_{0,1} && w_{0,2} && w_{0,3} && w_{0,4} && w_{0,5} \\
w_{1,0} && w_{1,1} && w_{1,2} && w_{1,3} && w_{1,4} && w_{1,5} \\ 
w_{2,0} && w_{2,1} && w_{2,2} && w_{2,3} && w_{2,4} && w_{2,5} \\
w_{3,0} && w_{3,1} && w_{3,2} && w_{3,3} && w_{3,4} && w_{3,5} \\
w_{4,0} && w_{4,1} && w_{4,2} && w_{4,3} && w_{4,4} && w_{4,5} \\
w_{5,0} && w_{5,1} && w_{5,2} && w_{5,3} && w_{5,4} && w_{5,5}\\
w_{6,0} && w_{6,1} && w_{6,2} && w_{6,3} && w_{6,4} && w_{6,5}\\
w_{7,0} && w_{7,1} && w_{7,2} && w_{7,3} && w_{7,4} && w_{7,5}\\
w_{8,0} && w_{8,1} && w_{8,2} && w_{8,3} && w_{8,4} && w_{8,5}\\
w_{9,0} && w_{9,1} && w_{9,2} && w_{9,3} && w_{9,4} && w_{9,5}
\end{pmatrix}; \textrm{Dimension=} 10 \times 6
$$

> Dimension of $$W_1$$ is $$10 \times 6$$ which is equivalent to: 
> $$\begin{pmatrix} \textrm{2nd dim of } X && \textrm{number  of nodes in hidden layer} \end{pmatrix}$$

> The values in this Weights Matrix is randomly initialised.

The randomly initialised matrix $$W_1$$ gets multiplied by the matrix $$X$$ to give a new matrix.

$$
H_1 =
X \cdot W_1 
$$

$$
H_1 =
\begin{pmatrix}
h_{0,0} && h_{0,1} && h_{0,2} && h_{0,3} && h_{0,4} && h_{0,5} \\
h_{1,0} && h_{1,1} && h_{1,2} && h_{1,3} && h_{1,4} && h_{1,5} \\
\vdots && \vdots && \vdots && \vdots && \vdots && \vdots \\
h_{n-1,0} && h_{n-1,1} && h_{n-1,2} && h_{n-1,3} && h_{n-1,4} && h_{n-1,5} \\
\end{pmatrix}; \textrm{Dimension=} n \times 6
$$

where each element

$$
h_{i,j}=\sum_{k=0}^{9}({x_{i,k} \times w_{k,j}});\ i=0 \ \textrm{to} \ n-1 ; j=0 \ \textrm{to} \ 5
$$

>Note: $$n$$ is the number of examples in the dataset.

#### Code

```python
# defining matrix multiplication function
def matMul(inp_: np.ndarray, W: np.ndarray):
  '''
  Performs Matrix Multiplication
  '''
  return np.dot(inp_, W)
```

### 2) Bias Addition
The output from *Matrix Multiplication* goes through the *bias addition* operation where a scalar value is added to each element.

For example, continuing with the *Matrix Multiplication* defined above, we'll have a *Bias vector* of length 6 (equal to the number of nodes in the hidden layer).
$$
B_1 =
\begin{pmatrix}
b_{0} && b_{1} && b_{2} && b_{3} && b_{4} && b_{5}
\end{pmatrix}; \textrm{Dimension=} 1 \times 6
$$

> In the case of Bias Addition, the matrix is *zero initialised*.

The zero initialised matrix is then added row by row to the input matrix $$H_1$$ (same bias vector is added to each example in the dataset, i.e. each row of matrix $$H_1$$ gets incremented by same amount).
$$
Z_1 =
H_1 + B_1 
$$

$$
Z_1 =
\begin{pmatrix}
z_{0,0} && z_{0,1} && z_{0,2} && z_{0,3} && z_{0,4} && z_{0,5} \\
z_{1,0} && z_{1,1} && z_{1,2} && z_{1,3} && z_{1,4} && z_{1,5} \\
\vdots && \vdots && \vdots && \vdots && \vdots && \vdots \\
z_{n-1,0} && z_{n-1,1} && z_{n-1,2} && z_{n-1,3} && z_{n-1,4} && z_{n-1,5} \\
\end{pmatrix}; \textrm{Dimension=} n \times 6
$$

where each element

$$
z_{i,j}={h_{i,j} + b_{j}};\ i=0 \ \textrm{to} \ n-1 ; j=0 \ \textrm{to} \ 5
$$

#### Code

```python
# Bias Addition function
def biasAdd(B: np.ndarray, inp_: np.ndarray):
  '''
  Performs Bias Addition
  '''
  return inp_ + B
```

### 3) Activation
Activation is an *elementwise operation* (which follows the *bias addition*), and it's main purpose is to add *non linearity*.

Let *$$f$$* be some continuous function, which is applied elemenwise to matrix $$Z_1$$. 
$$
A_1 = f(Z_1)
$$

$$
A_1 =
\begin{pmatrix}
a_{0,0} && a_{0,1} && a_{0,2} && a_{0,3} && a_{0,4} && a_{0,5} \\
a_{1,0} && a_{1,1} && a_{1,2} && a_{1,3} && a_{1,4} && a_{1,5} \\
\vdots && \vdots && \vdots && \vdots && \vdots && \vdots \\
a_{n-1,0} && a_{n-1,1} && a_{n-1,2} && a_{n-1,3} && a_{n-1,4} && a_{n-1,5} \\
\end{pmatrix}; \textrm{Dimension=} n \times 6
$$

where each element

$$
a_{i,j}=f(z_{i,j});\ i=0 \ \textrm{to} \ n-1 ; j=0 \ \textrm{to} \ 5
$$

There are several *Activation functions* like *Sigmoid* activation:

$$
\textrm{Sigmoid} =\frac{1}{1+e^{-Z}}
$$

#### Code

```python
def sigmoid(Z):
	'''
	Defining Sigmoid Activation
	'''
	return 1/(1+np.exp(-Z))
```

> Linear Activation is another example.

```python
def linear(Z):
	'''
	Defining Linear Activation
	'''
	return Z
```

## Forward Pass
Now that all the required operations are defined, let's put them together in correct order to form a complete *Neural Network*. The input features are then passed through the network to give predictions and this process is known as *Forward Pass*. 

![03.png](https://raw.githubusercontent.com/tgautam03/tgautam03.github.io/master/images/01-iota-intuitiveDL/03.png)


> Parameters are initialized before the forward pass, hence I've defined a separate fuction to do that.

```python
def init_weights(numX0, numH1, numH2):
    '''
    Initializing Parameters for the NN
		
		numX0: number of features in input dataset
		numH1: number of nodes in hidden layer
		numH2: number of nodes in output layer
    '''
    # defining weights and biases
    np.random.seed(0)
    W1 = np.random.rand(numX0,numH1)*0.01
    B1 = np.zeros((1, numH1))

    np.random.seed(1)
    W2 = np.random.rand(numH1, numH2)*0.01
    B2 = np.zeros((1, numH2))

    params = [W1, B1, W2, B2]
    return params
```
```python
def forward(inp_: np.ndarray,
            params: list):
    '''
    Forward pass through a Neural Network
    '''
    W1, B1, W2, B2 = params

    # Forward Pass
    H1 = matMul(inp_, W1)
    Z1 = biasAdd(B1, H1)
    A1 = sigmoid(Z1)

    H2 = matMul(A1, W2)
    Z2 = biasAdd(B2, H2)
    A2 = linear(Z2)

    cache = (H1, Z1, A1, H2, Z2)
    return A2, cache
```

After running the above code we get the value for $$Yhat$$, but you'll notice that the corresponding value in ground truth is very different from this.

To get the network to predict accurately, we need to modify the *weights* and *biases* by looking at the difference between *predicted* and *ground-truth* values. This is done by the process called *Training* a Neural Network.

> Notice that along with the prediction $$Yhat$$, I'm also returning the intermediate values. You'll see later how all this fits into the *training* process.

## Loss Function
Before *updating the weights and biases*, we need to quantify the difference between predictions and ground-truth labels. For this purpose, a *Loss Function* is used. 

![04.png](https://raw.githubusercontent.com/tgautam03/tgautam03.github.io/master/images/01-iota-intuitiveDL/04.png)

### Mean Squared Error (MSE)
A MSE loss function computes the *average of the squares of differences* between the predictions and labels (ground truth).
$$
\textrm{MSE}=\frac{1}{n}\sum_{i=0}^{n-1}(Y_i-Yhat_i)^2;
$$

where:
- $$Y_i  \textrm{ : ground-truth }$$
- $$Yhat_i \textrm{ : prediction}$$
- $$n \textrm{ : number of examples}$$

#### Code
```python
def MSE(y, yHat):
    n = y.shape[0]
    return 1/n*np.sum((y-yHat)^2, axis=0)
```

The goal now is to minimize this value (ideally make it zero but that never happens in real life), which is done by updating the weights using an algorithm called *Gradient Descent*.


## Conclusion
- In this blog post, we saw how different *operations* are put together to form a Neural Network. 
- In the next post, I'll discuss how a Neural Network is trained to give meanful predictions (i.e. minimize the value of the loss function)

## Additional Material (recommended)
- [**3blue1brown's** Deep Learning Series](https://www.youtube.com/watch?v=aircAruvnKk)

## References
- Weidman, Seth. Deep Learning from Scratch: Building with Python from First Principles. " O'Reilly Media, Inc.", 2019.
- Trask, Andrew W. "Deep learning." (2019).