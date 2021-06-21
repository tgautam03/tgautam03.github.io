---
layout: post
title:  "Part 2: Training a Neural Network"
date:   2021-06-21 15:46:08 +0530
categories: PyGrad
---

## Topics Covered
- How a Neural Network is trained.

## Introduction
In the last post, we saw how a Neural Network is set-up, and we'll take a look at how it's parameters are updated such that we get good predictions. Apart from the *operations* view-point, we can also look at a Neural Network from a point of high abstraction. 

Consider a Neural Network as a *box* with four knobs (each representing a weight and bias), and the goal is to tune the knobs such that the output of the loss function is minimum (or zero).

![05.png](https://raw.githubusercontent.com/tgautam03/tgautam03.github.io/master/images/01-iota-intuitiveDL/05.png)

Now the question arises, should I turn the knobs left or right? and by how much? 
>This will get answered in the next section where I'll explain *backpropagation* and *gradient descent*.

## Backpropagation and Gradient Descent
The *direction* and *amount* by which a knob should be turned is given by the partial derivatives, and these partial derivatives are calculated using *backpropagation* (as you'll see later how we move from back to front). In our case we have four possible knobs or parameters (one for each weight and bias), and there's a partial derivative associated with each i.e. $$\frac{dLoss}{dW_1}$$, $$\frac{dLoss}{dW_2}$$, $$\frac{dLoss}{db_1}$$, $$\frac{dLoss}{db_2}$$

All of these partial derivatives are easy to compute using *chain rule* as follows:

1) $$ \frac{dLoss}{dW_2}=\frac{dLoss}{dYhat} \times \frac{dYhat}{dZ_2} \times \frac{dZ_2}{dH_2} \times \frac{dH_2}{dW_2}$$

2) $$\frac{dLoss}{dB_2}=\frac{dLoss}{dYhat} \times \frac{dYhat}{dZ_2} \times \frac{dZ_2}{dB_2}$$

3) $$\frac{dLoss}{dW_1}=\frac{dLoss}{dYhat} \times \frac{dYhat}{dZ_2} \times \frac{dZ_2}{dH_2} \times \frac{dH_2}{dA_1} \times \frac{dA_1}{dZ_1} \times \frac{dZ_1}{dH_1} \times \frac{dH_1}{dW_1}$$

4) $$\frac{dLoss}{dB_1}=\frac{dLoss}{dYhat} \times \frac{dYhat}{dZ_2} \times \frac{dZ_2}{dH_2} \times \frac{dH_2}{dA_1} \times \frac{dA_1}{dZ_1} \times \frac{dZ_1}{dB_1}$$

These equations look quite daunting, but in reality they're very easy to calculate if we break them up into multiple parts. Let's now start from the tail-end of out Network (i.e. loss function layer) and move towards input layer.

### Gradient of Loss Function

![06.png](https://raw.githubusercontent.com/tgautam03/tgautam03.github.io/master/images/01-iota-intuitiveDL/06.png)

In the above figure, *blue* arrow indicates *forward pass* and *red* arrow indicates *backward pass* where the derivative of Loss Function with respect to prediction ($$\frac{dLoss}{dYhat}$$) is stored in variable `dYhat`.

If the loss function is defined as: 

$$
\textrm{MSE}=\frac{1}{n}\sum_{i=0}^{n-1}(Y_i-Yhat_i)^2
$$

Then from calculus, we know that: 
$$
\begin{aligned}
\frac{dLoss}{dYhat} =
\begin{pmatrix}
\frac{dLoss}{dYhat^{(0)}} \\
\frac{dLoss}{dYhat^{(1)}} \\
\vdots \\
\frac{dLoss}{dYhat^{(n-1)}} \\
\end{pmatrix}
\end{aligned}
$$

where each element

$$
\frac{dLoss}{dYhat^{(i)}}=-\frac{2}{n}\times (Y_i-Yhat_i)  
$$

$$Y_i \textrm{ and } Yhat_i \textrm{ are the label and prediction from the } i^{th} \textrm{ example} $$
 

#### Code

```python
def gradMSE(y, yHat):
    '''
    Function that computes gradient
    of loss wrt yHat
    '''
    n = y.shape[0]
    return -2/n*(y-yHat)
```

### Gradient of Linear Activation
In the Neural Network architecture, we have *Linear Activation* layer behind the *loss function*.

![07.png](https://raw.githubusercontent.com/tgautam03/tgautam03.github.io/master/images/01-iota-intuitiveDL/07.png)


From chain rule, we know that:
$$\frac{dLoss}{dZ_2}=\frac{dLoss}{dYhat} \times \frac{dYhat}{dZ_2}$$

and

$$
\frac{dYhat}{dZ_2}=
\begin{pmatrix}
1\\
1\\
\vdots \\
1\\
\end{pmatrix}; \textrm{Dimension}=n \times 1
$$

> Note: Dimension of $$\frac{dLoss}{dZ_2}$$ is same as dimension of $$Z_2$$.

As we already have computed $$\frac{dLoss}{dYhat}$$, we pass that into the function that computes the gradient of Linear Activation with respect to its inputs ($$\frac{dYhat}{dZ_2}$$), which in turn returns gradient of Loss with respect to the inputs of the Linear Activation (i.e. $$\frac{dLoss}{dZ_2}$$).

> During forward pass, *Linear Activation* was applied elemenwise, hence during backward pass $$\frac{dLoss}{dYhat}$$ will also get multiplied elementwise to $$\frac{dYhat}{dZ_2}$$.

#### Code

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

### Gradient of Bias Addition



![08.png](https://raw.githubusercontent.com/tgautam03/tgautam03.github.io/master/images/01-iota-intuitiveDL/08.png)



The Bias Addition layer is a bit different from *Loss* or any other *Activation layer*. This layer has two different gradients ($$\frac{dZ_2}{dB2}$$ and $$\frac{dZ_2}{dH2}$$), but they're computed the same way. 

Using chain rule:
$$\frac{dLoss}{dB_2}=\frac{dLoss}{dZ_2} \times \frac{dZ_2}{dB_2}$$
$$\frac{dLoss}{dH_2}=\frac{dLoss}{dZ_2} \times \frac{dZ_2}{dH_2}$$

and

- $$\frac{dZ_2}{dB_2}=\sum_{i=0}^{n-1}\frac{dZ_2^{(i)}}{dB_2};\ \textrm{where }\frac{dZ_2^{(i)}}{dB_2}\ \textrm{belongs to } i^{th} \textrm{example}$$ 
- $$\frac{dZ_2}{dH_2}=\begin{pmatrix}1\\1\\\vdots \\1\\\end{pmatrix}; \textrm{Dimension}=n \times 1$$ 

> Similar to *Linear Activation*, during forward pass the bias values are added elementwise to inputs hence the same elementwise product will be done during backward pass. 

> Notice that while computing $$\frac{dLoss}{dB_2}$$, we are summing along the axis which contains different examples because during forward pass same set of bias values are added to each example.

#### Code

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

### Gradient of Matrix Multiplication


![09.png](https://raw.githubusercontent.com/tgautam03/tgautam03.github.io/master/images/01-iota-intuitiveDL/09.png)



The Matrix Multiplication layer is very similar to Bias Addition Layer. This layer also has two different gradients ($$\frac{dH_2}{dW2}$$ and $$\frac{dH_2}{dA1}$$), but they're computed in a different way. 

Using chain rule:
$$\frac{dLoss}{dW_2}=\frac{dLoss}{dH_2} \times \frac{dH_2}{dW_2}$$
$$\frac{dLoss}{dA_1}=\frac{dLoss}{dH_2} \times \frac{dH_2}{dA_1}$$

and we know that, if $$u=v \cdot w$$, then $$\frac{du}{dv}=w^T$$ and $$\frac{du}{dw}=v^T$$, so in our case

- $$\frac{dH_2}{dW_2}=A_1^T$$ 
- $$\frac{dH_2}{dA_1}=W_2^T$$ 

One thing to keep in mind here is that there's **dot product** used instead of elementwise product. Hence, 
- $$\frac{dLoss}{dW_2}=A_1^T \cdot \frac{dLoss}{dH_2}$$
- $$\frac{dLoss}{dA_1}=\frac{dLoss}{dH_2} \cdot W_2^T$$

> I used to get confused all the time as to whether I should pre or post multiply the $$\frac{dLoss}{dH_2}$$ term? 

It's easy (but lengthy) to derive, but I've a trick (not trick but an easy way to remember) that'll answer the above question.
Note that in the forward pass i.e. $$H_2=A_1 \cdot W_2$$, if the matrix is pre-multiplied in the dot product (i.e. $$A_1$$),  then in $$\frac{dLoss}{dA_1}$$, $$\frac{dLoss}{dH_2}$$ will also get pre-multiplied. Althernately, if the matrix is post-multiplied in the dot product (i.e. $$W_2$$),  then in $$\frac{dLoss}{dW_2}$$, $$\frac{dLoss}{dH_2}$$ will also get post-multiplied.

#### Code

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
### Gradient of Sigmoid Activation

![10.png](https://raw.githubusercontent.com/tgautam03/tgautam03.github.io/master/images/01-iota-intuitiveDL/10.png)

Sigmoid Activation layer exactly the same as Linear Activation, except that the gradient of Sigmoid with respect to its input is non-zero.

From chain rule, we know that:
$$\frac{dLoss}{dZ_1}=\frac{dLoss}{dA1} \times \frac{dA1}{dZ_1}$$

and

$$\frac{dA1}{dZ_1}=\textrm{sigmoid}(Z_1)\times(1-\textrm{sigmoid}(Z_1))$$

> Note: Elementwise product of gradients again.

#### Code

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

## Backward Pass
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

## Gradient Descent
After computing the gradients of loss with respect to parameters, lets update the initialially randomised parameters. We use a small step length `alpha` to ensure that the new parameters are definintely better than the previous ones (there's risk of overshooting the ideal parameters will `alpha` is too large).

1) $$W_2 = W_2 - \alpha \times \frac{dLoss}{dW_2}$$

2) $$B_2 = B_2 - \alpha \times \frac{dLoss}{dB_2}$$

3) $$W_1 = W_1 - \alpha \times \frac{dLoss}{dW_1}$$

4) $$B_1 = B_1 - \alpha \times \frac{dLoss}{dB_1}$$

#### Code

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

## Conclusion
- In this blog post, we saw that using gradient descent we can get the *ideal parameters* from any random start point. 
- In the next post, I'll put everything together (compactly) and discuss how we can efficiently train the same Neural Network on many training examples and then make predictions on unseen examples.
- I'll also discuss the challenges that we'll face if there's a need to expand the Neural Network beyond 2 layers.

## Additional Material (recommended)
- [**3blue1brown's** Deep Learning Series](https://www.youtube.com/watch?v=aircAruvnKk)

## References
- Weidman, Seth. Deep Learning from Scratch: Building with Python from First Principles. " O'Reilly Media, Inc.", 2019.
- Trask, Andrew W. "Deep learning." (2019).




