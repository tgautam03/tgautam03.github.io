---
layout: post
title:  "Part 3: Predicting the Disease Progression"
date:   2021-06-21 18:50:08 +0530
categories: PyGrad
---

## Topics Covered
- Putting together *forward pass*, *loss function* and *backward pass*.
- Making predictions on the test set.
- Discussing the challenges if there's a need to alter the Neural Network architecture.

## Introduction
In the last post, we saw how a Neural Network is trained. Before proceeding to predictions let's put everything we've done so far together and train the Neural Network for multiple epochs. 

A *epoch* is a single *forward* and *backward* pass through the complete dataset. The parameters are updated at the end of each epoch. After the defined epochs are completed, the updated parameters are stored and used to make predictions on unseen dataset.

## Training Neural Network

```python
def train(X, Y, epochs, alpha):
    '''
    A function that performs:
    1) Forward pass
    2) Calculates loss
    3) Backward pass
    4) Update Weights

    X: Training Input data
    Y: Training Labels
    epochs: Number of epochs
    '''

    # list to store loss after each epoch
    lossVals = []

    # initialising parameters
    params = init_weights(X.shape[1], 6, Y.shape[1])

    for i in range(epochs):
        # forward pass
        yHat, cache = forward(X, params)

        # loss
        lossVal = MSE(Y, yHat)

        # backward pass
        param_grads = backward(X, Y, yHat, params, cache)

        # updating weights
        params = update(params, param_grads, alpha)

        # storing the loss value
        lossVals.append(lossVal)

        # printing the loss value
        if i%1000 == 0:
            print("Loss: ", lossVal)

    return params, lossVals
```

Now that we've defined the function to *train* a Neural Network, let's put to work. Before starting the training process, we have to split the dataset into *training set* and *testing set*. The purpose of this is to make sure that the Network has not seen a few examples while training, which we'll then use at the last to judge its performance.

We'll do this with the help of [scikit-learn's](https://scikit-learn.org/stable/index.html) `train_test_split` module.
```python
# importing scikit-learn's train test split module
from sklearn.model_selection import train_test_split

# splitting training and testing dataset with 70-30 ratio
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=50)
```

Let''s now run the `train` function with `alpha` as 0.01 and 10000 epochs.
```python
params, lossVals = train(X_train, y_train, 10000, 0.01)
```

After running the above code, we get a loss of 2497.04 (which isn't quite good!!).

We can visualise the loss curve using [matplotlib]().

```python
import matplotlib.pyplot as plt

plt.title("Loss Value vs epoch")
plt.plot(lossVals)
plt.xlabel("#epoch")
plt.ylabel("Loss Value")
plt.show()
```

![12.png](:/b973fecb875742f69a2bdfbc91fd4d50)


The curve above shows that the network isn't improving much after a few thousand epochs. Let's see how the test results look.

## Testing the trained Neural Network

During testing, the updated parameters are fed into the forward pass and the Mean Squared Error is calculated on the test set.

```python
def test(X, Y, params):
    '''
    A function that performs:
    1) Forward pass
    2) Calculates loss

    X: Training Input data
    Y: Training Labels
    params: Parameters of trained NN
    '''

    # forward pass
    yHat, _ = forward(X, params)

    # loss
    lossVal = MSE(Y, yHat)

    print("Mean Squared Error on test-set: ", lossVal)

    return lossVal
```
```python
testLossVal = test(X_test, y_test, params)
```
After running the above snippet,  we get the MSE on test set as 2453.26. 

This is not a good result, as this means we are on an average 49.5 away from desired prediction. Perhaps we can do something to improve the results?

## Improving the Neural Network
There are several things that can be done to improve the results, and one of them is to increase the number of layers. To do so, we need to modify the following functions:
1) `init_weights`
2) `forward`
3) `backward`
4) `train`

Now you might have sensed a problem here. This needs too much modifying and is prone to error. 

> You can try to add more layers to this code and you'll realise how confusing it is!!

## Conclusions
-  It's fairly easy to code a Neural Network without any abstractions, but this does not scale well.
-  Prototyping is almost impossible with this code structure.
-  Next, I'll discuss an alternative way to build a Deep Learning library, where I'll use several layers of abstractions and hence making the final code structure very similar to advanced libraries like [TensorFlow](https://www.tensorflow.org/).

## Additional Material (recommended)
- [**3blue1brown's** Deep Learning Series](https://www.youtube.com/watch?v=aircAruvnKk)

## References
- Weidman, Seth. Deep Learning from Scratch: Building with Python from First Principles. " O'Reilly Media, Inc.", 2019.
- Trask, Andrew W. "Deep learning." (2019).