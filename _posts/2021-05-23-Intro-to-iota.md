---
layout: post
title:  "iota: A Deep Learning framework"
date:   2021-05-23 18:46:08 +0530
categories: py
---

## Topics Covered

- The Motivation behind creating a Deep Learning Library.
- The Basics of Deep Learning.
- A Broad structure of the library that I'll build.


## Why am I building a Deep Learning Framework?
> In a world with great frameworks like PyTorch, TensorFlow, etc, is it even necessary to know how to build a framework? 

The short answer to this is, **ABSOLUTELY NOT!!!**. You can do amazing work without knowing how to code a Deep Learning library, but I feel that it's important to know the low-level stuff, because it'll flatten the learning curve when you start with a framework of your choice. If you have an idea of how a Deep Learning Framework is written, everything will seem intuitive and you'll pick up the syntax/concepts very easily.

> With *whys* regarding the iota (yes, I'm calling my framework iota) out of the way, let's dive into *how* I'm going to do this and *what* is required for this project. 

To build *iota*, I'll be using no third-party libraries except [Numpy](https://numpy.org/) (for speed and basic matrix operations) and [Matplotlib](https://matplotlib.org/) (for plotting). For testing the code, I'll use [PyTorch](https://pytorch.org/), but it'll not be used during development in any way other than checking the correctness of the algorithms.

I'll post a series of blog posts that will take you through this journey and I hope you'll learn something out of this. You can also follow the project on my GitHub account [here](https://github.com/tgautam03/iota) and stay updated.

> Note: In this series, I'll not be going deep into the mathematics (as there are several excellent resources exactly for this), instead this is for someone like me who's more comfortable with the Maths side of things but always wondered how it gets translated into efficient code.

## Overview of *Deep Learning*
In this section, I'll put forth the **big picture** and layout of how Deep Learning works under the hood. I'll use this as the basis to design the library. 

### Picture
Let's consider a simple two-layered Deep Neural Network shown below, where * grey circles* represent neurons.

![00.png](https://raw.githubusercontent.com/tgautam03/tgautam03.github.io/master/images/01-iota/00.png)

Notice that there are two types of *arrows* in the figure (green & red), and each represent a different operation:
- *Green Arrows*: Information from all neurons are getting mixed. This is a matrix dot product (as the whole) in the case of Neural Network.
- *Red Arrows*: No mixing of information between neurons. This is an elementwise operation on each neuron and is known as *activation* in the case of Neural Network.

### Maths
We can easily translate this picture into equations. Let's consider the three neurons in the *Input Layer* as $x_1, x_2, x_3$. We can represent these three values in a $$1 \times 3$$ matrix $$X$$ given as:

$$ 
X = 
\begin{pmatrix}
x_1 & x_2 & x_3
\end{pmatrix}
$$

The next step is to represent the *black arrows* operation, which is the post multiplication of matrix $$X$$ by a matrix $$W_1$$ given as:

$$
W_1 = 
\begin{pmatrix}
w_{11}^1 & w_{12}^1 & w_{13}^1 & w_{14}^1 \\
w_{21}^1 & w_{22}^1 & w_{23}^1 & w_{24}^1 \\
w_{31}^1 & w_{32}^1 & w_{33}^1 & w_{34}^1 \\

\end{pmatrix}
$$

The output of this matrix multiplication is said to be $$Z_1$$:

$$ 
Z_1 = 
\begin{pmatrix}
x_1 & x_2 & x_3
\end{pmatrix}
\begin{pmatrix}
w_{11}^1 & w_{12}^1 & w_{13}^1 & w_{14}^1 \\
w_{21}^1 & w_{22}^1 & w_{23}^1 & w_{24}^1 \\
w_{31}^1 & w_{32}^1 & w_{33}^1 & w_{34}^1 \\
\end{pmatrix}
$$

$$
Z_1 =
\begin{pmatrix}
\sum_{i=1}^{i=3} x_iw_{i1}^1 & \sum_{i=1}^{i=3} x_iw_{i2}^1 & \sum_{i=1}^{i=3} x_iw_{i3}^1 & \sum_{i=1}^{i=3} x_iw_{i4}^1
\end{pmatrix}
$$

$$
Z_1 =
\begin{pmatrix}
z_1^1 & z_2^1 & z_3^1 & z_4^1
\end{pmatrix}; z^1 \ \textrm{and} \ w_{ij}^1 \ \textrm{is not power by 1, but simple notation}
$$

After this, we see the *red arrows*, which represent elementwise operation on $$Z_1$$ and the output from this is written as $$A_1$$:

$$
A_1 = f()
\begin{pmatrix}
z_1^1 & z_2^1 & z_3^1 & z_4^1
\end{pmatrix}; 
\\ \textrm{where} \ f()\ \textrm{is any function that acts on elements of the matrix}\ Z_1
$$

$$
A_1 =
\begin{pmatrix}
f(z_1^1) & f(z_2^1) & f(z_3^1) & f(z_4^1)
\end{pmatrix}
=
\begin{pmatrix}
a_1 & a_2 & a_3 & a_4
\end{pmatrix}
$$

The final operation is again the matrix multiplication by $$W_2$$ and the output is given as $$Z_2$$:

$$ 
Z_2 = 
\begin{pmatrix}
a_1 & a_2 & a_3 & a_4
\end{pmatrix}
\begin{pmatrix}
w_{11}^2 & w_{12}^2 \\
w_{21}^2 & w_{22}^2 \\
w_{31}^2 & w_{32}^2 \\
w_{41}^2 & w_{42}^2
\end{pmatrix}
$$

$$
Z_2 =
\begin{pmatrix}
z_1^2 & z_2^2 & z_3^2 & z_4^2
\end{pmatrix}; z^2 \ \textrm{and} \ w_{ij}^2 \ \textrm{is not squared, but simple notation}
$$

> Note: I've ignored the bias term.

The **learning** part in Deep Learning means that we **learn** the matrices $$W_1 \ \textrm{and} \ W_2$$ via an algorithm called gradient descent, which can be summarised in three steps:
- Start with random $$W_1$$ and $$W_2$$.
- Compare $$Z_2$$ with ground truth data
- Adjust $$W_1$$ and $$W_2$$ such that $$Z_2$$ is closest to ground-truth label.

> Note: I'll explain the *learning process* in detail in the very next section.

### Putting everything together

![01.png](https://raw.githubusercontent.com/tgautam03/tgautam03.github.io/master/images/01-iota/01.png)

In the diagram above, we can see how with the help of different *operations* (marked in the green box), the neurons interact with each other.  

The whole process of *training* a Neural Network (i.e. learning $$W_1$$ and $$W_2$$) is subdivided into four steps:

- Forward Pass
- Computing the loss
- Backward Pass
- Updating Parameters

#### Forward Pass
This can be summarized in just three equations:

$$
Z_1 = X \cdot W_1 
$$

$$
A_1 = f(Z_1)
$$

$$
Z_2 = A_1 \cdot W_2
$$

#### Computing the loss
Now once we have $$Z_2$$, the next step is to compare this output to some ground-truth label $$Y$$. A function called *loss function* takes in both $$Z_2$$ and $$Y$$ as inputs and maps them into a single number that represents how similar or different they are. For example, Mean Squared Error function: $$\sqrt(Z_2^2 - Y^2)$$.

For the sake of generality let's consider the loss function to be $$L(Y, Z_2)$$, hence

$$
lossVal = L(Y,Z_2)
$$

#### Backward pass
With the help of backward pass, we update $W_1$ and $W_2$ such that $lossVal$ is minimized. Before we update the parameters, we have to find the number that we have to add or subtract to decrease the $lossVal$? 

This is done with the help of chain rule from calculus. We compute the derivative of $lossVal$ with respect to $W_1$ and $W_2$ and then update them with this quantity. 

$$
\frac{d\ lossVal}{d\ W_2} = \frac{d\ lossVal}{d\ Z_2} \times \frac{d\ Z_2}{d\ W_2}
$$

$$
\frac{d\ lossVal}{d\ W_2} = A_1^T \cdot L^\prime; \textrm{where} \cdot \ \textrm{is dot product and } \ A_1^T \textrm{is the transpose}
$$

> I used to get confused all the time whether I should pre-multiply or post-multiply $$A_1^T$$? 
> It's fairly easy (but lengthy) to show its derivation but an easier way to remember this is that if in the equation $$Z_2 = A_1 \cdot W_2$$, $$A_1$$ is pre-multiplied, then in the derivative the $$A_1^T$$ will also get pre-multiplied.

We now have a derivative of loss with respect to $$W_2$$, and to get to $$W_1$$, we have to move through activation function $$f()$$.

$$
\frac{d\ lossVal}{d\ W_1} = \frac{d\ lossVal}{d\ Z_2} \times \frac{d\ Z_2}{d\ A_1} \times \frac{d\ A_1}{d\ Z_1} \times \frac{d\ Z_1}{d\ W_1}
$$

Splitting the above equation into two products and solving similarly

$$
\frac{d\ lossVal}{d\ W_1} = \frac{d\ lossVal}{d\ Z_1} \times \frac{d\ Z_1}{d\ W_1}
$$

$$
\frac{d\ lossVal}{d\ W_1} = X^T \cdot \frac{d\ lossVal}{d\ Z_1}
$$



Let's look at the $$\frac{d\ lossVal}{d\ Z_1}$$ now where we can write this partial derivative as

$$
\frac{d\ lossVal}{d\ Z_1} = ((L^\prime \cdot W_2^T) \times f^\prime);\textrm{where} \cdot \ \textrm{is dot product and } \times \textrm{is the elementwise product}
$$
> Note: Again, the equation was $$Z_2 = A_1 \cdot W_2$$, where $$W_2$$ was post-multiplied hence $$W_2^T$$ will also get post multiplied. For the case of activation, it was an element-wise operation hence the derivate product will also be elementwise.

Now combining the above two results, we get

$$
\frac{d\ lossVal}{d\ W_1} = X^T \cdot ((L^\prime \cdot W_2^T) \times f^\prime)
$$

#### Updating Parameters
Finally, the parameters are updated with will reduce the loss function value.

$$
W_1 = W_1 - \frac{d\ lossVal}{d\ W_1}
$$

$$
W_2 = W_2 - \frac{d\ lossVal}{d\ W_2}
$$


## Structure of iota
The mathematics behind Deep Learning is fairly simple, and we can easily write the above two-layered Neural Network in python code under 20 lines. The problem arises when we need very large Neural Networks with hundreds of layers and millions of parameters. So, how can we code such a network where with minimum effort, we can scale the Network to whatever size we need or whatever activations or even layers?

To answer all the above questions, I'll move to another diagram of our Network where I'll explain abstractions on different levels. Let's first look at the lowest level where I can divide the whole network into *operations* of two types:
- Operations with Parameters:  Dot product
- Operations without Parameters: Activation Function

The above explanation is put into a diagram shown below:

![02.png](https://raw.githubusercontent.com/tgautam03/tgautam03.github.io/master/images/01-iota/02.png)

Now that I have operations, I can group multiple operations into a higher abstraction called *Layers*. The motivation behind doing this is that it'll be tedious to work with individual operations and instead what I can do is put multiple of these into one layer and then work with the layers individually.

![03.png](https://raw.githubusercontent.com/tgautam03/tgautam03.github.io/master/images/01-iota/03.png)

> Note: In our example, the 2nd layer has *Linear* as the element-wise operation.

The final abstraction would be the whole *Neural Network* which is made up of multiple Layers. 


![04.png](https://raw.githubusercontent.com/tgautam03/tgautam03.github.io/master/images/01-iota/04.png)

## Conclusion
In this post, I first covered my motivation for creating a Deep Learning Framework from scratch, then I discussed briefly the mathematics behind Deep Learning and finally laid out the broad strokes with which I'll design my library.

In coming posts, I'll delve deep into the different parts like Operations, Layers, Neural Network, etc and write efficient code that can be scaled properly. 

> If you're still confused about the last section, please stick around for further blog posts and everything will get clearer as I discuss them in detail.

## Additional Material (recommended)
- [**3blue1brown's** Deep Learning Series](https://www.youtube.com/watch?v=aircAruvnKk)

## References
- Weidman, Seth. Deep Learning from Scratch: Building with Python from First Principles. " O'Reilly Media, Inc.", 2019.
- Trask, Andrew W. "Deep learning." (2019).

