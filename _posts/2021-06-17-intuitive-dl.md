---
layout: post
title:  "An Intuitive (yet Practical) overview of Deep Learning"
date:   2021-06-17 12:46:08 +0530
categories: py
---

## Topics Covered
- The Basics of Deep Learning.
- Building a Simple Neural Network from Scratch

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

For the sake of simplicity, let's first look at one example (the very first) from the dataset. 
$$ 
\begin{aligned}
X^{(0)T} =
\begin{pmatrix}
x_{1} \\
x_{2} \\ 
x_{3} \\
x_{4} \\
x_{5} \\
x_{6}\\
x_{7}\\
x_{8}\\
x_{9}\\
x_{10}
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

```python
# Extracting the first example
X0 = X[0].reshape(1, 10)
```

The various *operations* used throughout the Deep Neural Network are described as follows:
### 1) Matrix Multiplication
During this operation, the input is multiplied by a *weights* matrix whose dimension is defined in accordance with the inputs and the desired output size. For Example in case of the very first layer which maps input vector $$X$$ (of length 10) to first hidden units vector $$H_1$$ (of length 6), the *weights* matrix is defined as

$$ 
W_1 =
\begin{pmatrix}
w_{1,1} && w_{1,2} && w_{1,3} && w_{1,4} && w_{1,5} && w_{1,6} \\
w_{2,1} && w_{2,2} && w_{2,3} && w_{2,4} && w_{2,5} && w_{2,6} \\ 
w_{3,1} && w_{3,2} && w_{3,3} && w_{3,4} && w_{3,5} && w_{3,6} \\
w_{4,1} && w_{4,2} && w_{4,3} && w_{4,4} && w_{4,5} && w_{4,6} \\
w_{5,1} && w_{5,2} && w_{5,3} && w_{5,4} && w_{5,5} && w_{5,6} \\
w_{6,1} && w_{6,2} && w_{6,3} && w_{6,4} && w_{6,5} && w_{6,6}\\
w_{7,1} && w_{7,2} && w_{7,3} && w_{7,4} && w_{7,5} && w_{7,6}\\
w_{8,1} && w_{8,2} && w_{8,3} && w_{8,4} && w_{8,5} && w_{8,6}\\
w_{9,1} && w_{9,2} && w_{9,3} && w_{9,4} && w_{9,5} && w_{9,6}\\
w_{10,1} && w_{10,2} && w_{10,3} && w_{10,4} && w_{10,5} && w_{10,6}
\end{pmatrix}; \textrm{Dimension=} 10 \times 6
$$

> Dimension of $$W_1$$ is $$10 \times 6$$ which is equivalent to: 
> $$\begin{pmatrix} \textrm{2nd dim of } X && \textrm{number  of nodes in hidden layer} \end{pmatrix}$$

#### Mathematics
Firstly, the matrix $$W_1$$ is *randomly initialised* and then it gets multiplied by the inputs to give a new matrix.

$$
H_1 =
X \cdot W_1 
$$

$$
H_1 =
\begin{pmatrix}
h_{1,1} && h_{1,2} && h_{1,3} && h_{1,4} && h_{1,5} && h_{1,6}
\end{pmatrix}; \textrm{Dimension=} 1 \times 6
\\
\textrm{where} \ h_{1,j}=\sum_{i=1}^{10}({x_i \times w_{i,j}}); j=1 \ \textrm{to} \ 6.
$$

```python
# defining matrix multiplication function
def matMul(inp_: np.ndarray, W: np.ndarray):
  '''
  Performs Matrix Multiplication
  '''
  return np.dot(inp_, W)

# seeding to get consistent random values
np.random.seed(0)

# initializing W1
W1 = np.random.rand(10,6)*0.01

# matrix multiplication X0 . W1
H1 = matMul(X0, W1)
```

### 2) Bias Addition
The output from *Matrix Multiplication* goes through the *bias addition* operation where a scalar value is added to each element.
$$
B_1 =
\begin{pmatrix}
b_{1} && b_{2} && b_{3} && b_{4} && b_{5} && b_{6}
\end{pmatrix}; \textrm{Dimension=} 1 \times 6
$$

#### Mathematics
In the case of Bias Addition, the matrix is *zero initialised*.
- Initialising bias:
$$
B_1 =
\begin{pmatrix}
0 && 0 && 0 && 0 && 0 && 0
\end{pmatrix}; \textrm{Dimension=} 1 \times 6
$$

- Bias Addition:
$$
Z_1 =
H_1 + B_1 
$$

$$
Z_1 =
\begin{pmatrix}
z_{1,1} && z_{1,2} && z_{1,3} && z_{1,4} && z_{1,5} && z_{1,6}
\end{pmatrix}; \textrm{Dimension=} 1 \times 6
$$

$$
\textrm{where} \ z_{1,j}={h_{1,j} + b_{j}};\ j=1 \ \textrm{to} \ 6.
$$


```python
# Bias Addition function
def biasAdd(B: np.ndarray, inp_: np.ndarray):
  '''
  Performs Bias Addition
  '''
  return inp_ + B

# initialising b1
B1 = np.zeros((1, 6))

# bias addition
Z1 = biasAdd(B1, H1)
```

### 3) Activation
Another *elementwise operation* follows the *bias addition*, and it's main purpose is to add *non linearity*.
#### Mathematics
Let's take an example of *Sigmoid* activation function: 
$$
\textrm{Sigmoid} =\frac{1}{1+e^{-Z}}
$$

Now, applying Sigmoid Activation to $$Z_1$$ gives $$A_1$$:
$$
A_1 = \textrm{sigmoid}(Z_1)
$$

$$
A_1 =
\begin{pmatrix}
a_{1,1} && a_{1,2} && a_{1,3} && a_{1,4} && a_{1,5} && a_{1,6}
\end{pmatrix}; \textrm{Dimension=} 1 \times 6
$$
$$
\textrm{where} \ a_{1,j}=\textrm{sigmoid}({z_{1,j}});\ j=1 \ \textrm{to} \ 6.
$$

```python
def sigmoid(Z):
	'''
	Defining Sigmoid Activation
	'''
	return 1/(1+np.exp(-Z))

# applying activation to Z1
A1 = sigmoid(Z1)
```

> Linear Activation is another example which is defined as follows:
```python
def linear(Z):
	'''
	Defining Linear Activation
	'''
	return Z
```

## Forward Pass
Once all the required operations are defined, they're put together in correct order to form a complete *Neural Network*. The input features are then passed through the network to give predictions and this process is known as *Forward Pass*. 

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
```python
# initializing parameters
params = init_weights(X0.shape[1], 6, 1)

# performing forward pass
Yhat, cache = forward(X0)
```

After running the above code we get the value for $$Yhat=0.00839$$, but the corresponding value in ground truth is $$151$$, which is not surprising as the initialized weights were very small.

To get the network to predict accurately, we need to modify the *weights* and *biases* by looking at the difference between *predicted* and *ground-truth* values. This is done by the process called *Training* a Neural Network.

> Notice that along with the prediction $$Yhat$$, I'm also returning the intermediate values. You'll see later how all this fits into the *training* process.

## Loss Function
Before *updating the weights and biases*, we need to quantify the difference between prediction and ground-truth label. For this purpose, a *Loss Function* is used. 

![04.png](https://raw.githubusercontent.com/tgautam03/tgautam03.github.io/master/images/01-iota-intuitiveDL/04.png)

### Mean Squared Error (MSE)
A MSE loss function computes the *average of the squares of differences* between the predictions and labels (ground truth).

#### Mathematics
$$
\textrm{MSE}=\frac{1}{n}\sum_{i=1}^n(Y_i-Yhat_i)^2;
\\ 
\textrm{where}\ n \ \textrm{is the number of examples} 
$$

#### Code
```python
def MSE(y, yHat):
    n = y.shape[0]
    return 1/n*np.sum((y-yHat)^2, axis=0)
```

Computing the MSE on the prediction and the ground truth yield a value of $$22798.46$$! The goal now is to minimize this value (ideally make it zero but that never happens in real life), which is done by updating the weights using an algorithm called *Gradient Descent*.

## Neural Network Training
Another way to look at a Neural Network is from a point of high abstraction. Consider a Neural Network as a *box* with four knobs (each representing a weight and bias), and the goal is to tune the knobs such that the output of the loss function is minimum (or zero).

![05.png](https://raw.githubusercontent.com/tgautam03/tgautam03.github.io/master/images/01-iota-intuitiveDL/05.png)

Now a question arises, should I turn the knobs left or right? and by how much? 
This question will get answered in the next section where I'll explain *backpropagation* and *gradient descent*.

### Backpropagation and Gradient Descent
The *direction* and *amount* by which a knob should be turned is given by the partial derivatives, and these partial derivatives are calculated using *backpropagation* (as you'll see later how we move from back to front). In our case we have four possible knobs or parameters (one for each weight and bias), and there's a partial derivative associated with each:
- $$\frac{dLoss}{dW_1}$$ 
- $$\frac{dLoss}{dW_2}$$
- $$\frac{dLoss}{db_1}$$
- $$\frac{dLoss}{db_2}$$

All of these partial derivatives are easy to compute using *chain rule* as follows:

1) $$ \frac{dLoss}{dW_2}=\frac{dLoss}{dYhat} \times \frac{dYhat}{dZ_2} \times \frac{dZ_2}{dH_2} \times \frac{dH_2}{dW_2}$$

2) $$\frac{dLoss}{db_2}=\frac{dLoss}{dYhat} \times \frac{dYhat}{dZ_2} \times \frac{dZ_2}{db_2}$$

3) $$\frac{dLoss}{dW_1}=\frac{dLoss}{dYhat} \times \frac{dYhat}{dZ_2} \times \frac{dZ_2}{dH_2} \times \frac{dH_2}{dA_1} \times \frac{dA_1}{dZ_1} \times \frac{dZ_1}{dH_1} \times \frac{dH_1}{dW_1}$$

4) $$\frac{dLoss}{db_1}=\frac{dLoss}{dYhat} \times \frac{dYhat}{dZ_2} \times \frac{dZ_2}{dH_2} \times \frac{dH_2}{dA_1} \times \frac{dA_1}{dZ_1} \times \frac{dZ_1}{db_1}$$

These equations look quite daunting, but in reality they're very easy to calculate if we break them up into multiple parts. 

#### Gradient of Loss Function
Let's start with the last layer and look at the loss function. Here we'll analyse how the inputs affect the output value. 

![06.png](https://raw.githubusercontent.com/tgautam03/tgautam03.github.io/master/images/01-iota-intuitiveDL/06.png)

In the above figure, *blue* arrow indicates *forward pass* and *red* arrow indicates *backward pass* where the derivative of Loss Function with respect to prediction ($$\frac{dLoss}{dYhat}$$) is stored in variable `dYhat`. 

Let's see how *Loss Function Gradient* can be coded for *Mean Squared Error*.
```python
def gradMSE(y, yHat):
    '''
    Function that computes gradient
    of loss wrt yHat
    '''
    n = y.shape[0]
    return -2/n*(y-yHat)
```

#### Gradient of Linear Activation


![07.png](https://raw.githubusercontent.com/tgautam03/tgautam03.github.io/master/images/01-iota-intuitiveDL/07.png)



From chain rule, we know that:
$$\frac{dLoss}{dZ_2}=\frac{dLoss}{dYhat} \times \frac{dYhat}{dZ_2}$$

Note that we already have computed $$\frac{dLoss}{dYhat}$$, hence we pass that into the function that computes the gradient of Linear Activation with respect to its inputs ($$\frac{dYhat}{dZ_2}$$), which in turn returns gradient of Loss with respect to inputs to the Linear Activation ($$\frac{dLoss}{dZ_2}$$).

> During forward pass, *Linear Activation* was applied elemenwise, hence during backward pass $$\frac{dLoss}{dYhat}$$ will also get multiplied elementwise to $$\frac{dYhat}{dZ_2}$$.

```python
def grad_linear(outGrad, inp_):
	'''
	Function that computes gradient
	of loss wrt inp_
		
	outGrad: Gradient of Loss wrt Yhat
	inp_: Input to Linear Activation i.e. Z2
	'''
	# Grad of Yhat wrt to Z2 is a matrix of ones
	dinp_ = outGrad * np.ones(inp_.shape) 
	return dinp_
```

#### Gradient of Bias Addition



![08.png](https://raw.githubusercontent.com/tgautam03/tgautam03.github.io/master/images/01-iota-intuitiveDL/08.png)



The Bias Addition layer is a bit different from Loss or any other Activation layer. This layer has two different gradients ($$\frac{dZ_2}{dB2}$$ and $$\frac{dZ_2}{dH2}$$), but they're computed the same way. 

Using chain rule:
$$\frac{dLoss}{dB2}=\frac{dLoss}{dZ_2} \times \frac{dZ_2}{dB2}$$
$$\frac{dLoss}{dH2}=\frac{dLoss}{dZ_2} \times \frac{dZ_2}{dH2}$$

> Similar to *Linear Activation*, during forward pass the bias values are added elementwise to inputs hence the same elementwise product will be done during backward pass. 

> Notice that while computing $$\frac{dLoss}{dB2}$$, we are summing along the axis which contains different examples because during forward pass same set of bias values are added to each example.

```python
def grad_biasAdd(outGrad, B, inp_):
	'''
	Function that computes gradient
	of loss wrt B and H2
	
	outGrad: Gradient of Loss wrt Z2
	B: Bias parameter
	inp_: Input to Bias Addition i.e. H2
	'''
	# grad of loss wrt bias parameter 
	dB = np.sum(outGrad, axis=0)
	# grad of loss wrt H2 
	dinp_ = outGrad * np.ones(inp_.shape)
	return dB, dinp_
```

#### Gradient of Matrix Multiplication


![09.png](https://raw.githubusercontent.com/tgautam03/tgautam03.github.io/master/images/01-iota-intuitiveDL/09.png)



The Matrix Multiplication layer is very similar to Bias Addition Layer. This layer also has two different gradients ($$\frac{dH_2}{dW2}$$ and $$\frac{dH_2}{dA1}$$), but they're computed the same way. 

Using chain rule:
$$\frac{dLoss}{dW_2}=\frac{dLoss}{dH_2} \times \frac{dH_2}{dW_2}$$
$$\frac{dLoss}{dA_1}=\frac{dLoss}{dH_2} \times \frac{dH_2}{dA_1}$$

> One thing to keep in mind here is there's **dot product** used instead of elementwise product. 
> We know that, if $$u=v \cdot w$$, then $$\frac{du}{dv}=w^T$$ and $$\frac{du}{dw}=v^T$$, hence 
> - $$\frac{dLoss}{dW_2}=A_1^T \cdot \frac{dLoss}{dH_2}$$
> - $$\frac{dLoss}{dA_1}=\frac{dLoss}{dH_2} \cdot W_2^T$$

> I used to get confused all the time as to whether I should pre or post multiply the $$\frac{dLoss}{dH_2}$$ term? 
> It's easy (but lengthy) to derive, but I've a trick (not trick but an easy way to remember) that'll answer the above question. Note that in the forward pass i.e. $$H_2=A_1 \cdot W_2$$, if the matrix is pre-multiplied in the dot product (i.e. $$A_1$$),  then in $$\frac{dLoss}{dA_1}$$, $$\frac{dLoss}{dH_2}$$ will also get pre-multiplied. Althernately, if the matrix is post-multiplied in the dot product (i.e. $$W_2$$),  then in $$\frac{dLoss}{dW_2}$$, $$\frac{dLoss}{dH_2}$$ will also get post-multiplied.

```python
def grad_matMul(outGrad, W, inp_):
	'''
	Function that computes gradient
	of loss wrt W and A1
	
	outGrad: Gradient of Loss wrt H2
	W: Weight parameter
	inp_: Input to Matrix Multiplication i.e. A1
	'''
	# grad of loss wrt weight parameter 
	dW = np.dot(inp_.T, outGrad)
	# grad of loss wrt A1 
	dinp_ = np.dot(outGrad, W.T)
	return dW, dinp_
```
#### Gradient of Sigmoid Activation

![10.png](https://raw.githubusercontent.com/tgautam03/tgautam03.github.io/master/images/01-iota-intuitiveDL/10.png)

Sigmoid Activation layer exactly the same as Linear Activation, except that the gradient of Sigmoid with respect to its input is non-zero.

From chain rule, we know that:
$$\frac{dLoss}{dZ_1}=\frac{dLoss}{dA1} \times \frac{dA1}{dZ_1}$$

> Note: Elementwise product of gradients again.

```python
def grad_sigmoid(outGrad, inp_):
	'''
	Function that computes gradient
	of loss wrt inp_
		
	outGrad: Gradient of Loss wrt A1
	inp_: Input to Sigmoid Activation i.e. Z1
	'''
	# Grad of Yhat wrt to Z2 is a matrix of ones
	dinp_ = outGrad * sigmoid(inp_)*(1-sigmoid(inp_)) 
	return dinp_
```

### Backward Pass
Now that we've defined all the parts, lets put them together.

![11.png](https://raw.githubusercontent.com/tgautam03/tgautam03.github.io/master/images/01-iota-intuitiveDL/11.png)


The above figure shows the complete *forward* (blue part) and *backward* (red part) pass. We can see that *backward pass* starts from the tail of the Network and moves towards the start, using values computed during forward pass at different stages. This is the reason we store the intermediate values computed during forward pass.

#### Code
```python
def backward(x, y, yHat, params, cache):
    '''
    Does Backward pass through the Network 
    to compute all the gradients required
    '''
    H1, Z1, A1, H2, Z2 = cache
    W1, B1, W2, B2 = params

    dYhat = grad_MSE(y, yHat)
    
    dZ2 = grad_linear(dYhat, Z2)

    dB2, dH2 = grad_biasAdd(dZ2, B2, H2)
    assert B2.shape == dB2.shape

    dW2, dA1 = grad_matMul(dH2, W2, A1)
    assert W2.shape == dW2.shape

    dZ1 = grad_sigmoid(dA1, Z1)

    dB1, dH1 = grad_biasAdd(dZ1, B1, H1)
    assert B1.shape == dB1.shape

    dW1, _ = grad_matMul(dH1, W1, x)
    assert W1.shape == dW1.shape

    param_grads = (dW1, dB1, dW2, dB2)
    
	return param_grads
```
```python
param_grads = backward(X0, Y0, Yhat, params, cache)
```

### Gradient Descent
Now that we have gradients of loss with respect to parameters, lets update the initialially randomised parameters. We use a small step length `alpha` to ensure that the new parameters are definintely better than the previous ones (there's risk of overshooting the ideal parameters will `alpha` is too large).

```python
def update(params, param_grads, alpha):
    '''
    Updating the Parameters using SGD
    '''
    assert len(params) == len(param_grads)

    for i in range(len(params)):
        params[i] -= alpha*param_grads[i]

    return params
```
```python
# updated parameters
params = update(params, param_grads, 0.01)
```

In theory, our parameters should now predict better. Lets find out by running the following code:

```python
Yhat, _, _ = forward(X0, params)
lossVal2 = MSE(Y0, Yhat)
```

This gives the new loss value as $$20573.71$$, which is definitely better than previous value but still far off the ideal value of $$0$$.

## Conclusion
- In this blog post, we saw that using gradient descent we can get the *ideal parameters* from any random start point. 
- In the next post, I'll put everything together (compactly) and discuss how we can efficiently train the same Neural Network on many training examples and then make predictions on unseen examples.

## Additional Material (recommended)
- [**3blue1brown's** Deep Learning Series](https://www.youtube.com/watch?v=aircAruvnKk)

## References
- Weidman, Seth. Deep Learning from Scratch: Building with Python from First Principles. " O'Reilly Media, Inc.", 2019.
- Trask, Andrew W. "Deep learning." (2019).