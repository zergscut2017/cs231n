import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################



  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in xrange(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores)  #  prevents numerical instability
    correct_class_score = scores[y[i]]

    exp_sum = np.sum(np.exp(scores))
    loss += np.log(exp_sum) - correct_class_score

    dW[:, y[i]] -= X[i]
    for j in xrange(num_classes):
      dW[:,j] += (np.exp(scores[j]) / exp_sum) * X[i]

  loss /= num_train
  loss += 0.5 * reg * np.sum( W*W )
  dW /= num_train
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0] # X.shape[0] or y.shape[0] are both ok. shape[0] is the row numbers
  if X.shape[0] == y.shape[0]:
    print "x or y is both ok"
  print(np.shape(X))
  print(np.shape(y))
  print(np.size(y))
  print('**************num_train: ***********')
  print(num_train)
  scores = X.dot(W)
  correct_class_score = scores[np.arange(num_train), y].reshape(num_train, 1)
  print('**************correct_class_score: ***********')
  print(np.shape(scores[np.arange(num_train), y]))
  print(scores[np.arange(num_train), y])
  exp_sum = np.sum(np.exp(scores), axis=1).reshape(num_train, 1)
  print('**************exp_sum: ***********')
  print(np.shape(np.sum(np.exp(scores), axis=1)))
  print(np.sum(np.exp(scores), axis=1))
  loss += np.sum(np.log(exp_sum) - correct_class_score)
  loss /= num_train
  loss += 0.5*reg*np.sum(W*W)

  margin = np.exp(scores) / exp_sum
  margin[np.arange(num_train), y] += -1
  dW = X.T.dot(margin)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

