import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    self.params['W1'] = weight_scale * np.random.randn(num_filters, C,
                                                       filter_size,
                                                       filter_size)
    self.params['b1'] = np.zeros(num_filters)

    conv_stride = 1
    conv_pad = (filter_size - 1) / 2
    conv_H = (H - filter_size + 2 * conv_pad) / conv_stride + 1
    conv_W = (W - filter_size + 2 * conv_pad) / conv_stride + 1
    conv_C = num_filters
    pool_height = 2
    pool_width = 2
    pool_stride = 2

    pool_H = (conv_H - pool_height) / pool_stride + 1
    pool_W = (conv_W - pool_width) / pool_stride + 1
    pool_C = conv_C
    dim_output = pool_H * pool_W * pool_C
    print("dim_output = {}".format(dim_output))
    self.params['W2'] = weight_scale * np.random.randn(dim_output, hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)

    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b3'] = np.zeros(num_classes)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    conv_h, conv_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    affine1_h, affine1_cache = affine_relu_forward(conv_h, W2, b2)
    affine2_h, affine2_cache = affine_forward(affine1_h, W3, b3)

    scores = affine2_h
    if y is None:
      return scores
    
    loss, grads = 0, {}
    loss, dE = softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W3 ** 2))

    dx, dW3, db3 = affine_backward(dE, affine2_cache)
    dx, dW2, db2 = affine_relu_backward(dx, affine1_cache)
    dx, dW1, db1 = conv_relu_pool_backward(dx, conv_cache)


    dW3 += self.reg * W3
    dW2 += self.reg * W2
    dW1 += self.reg * W1

    grads['W3'] = dW3
    grads['b3'] = db3.flatten()

    grads['W2'] = dW2
    grads['b2'] = db2.flatten()

    grads['W1'] = dW1
    grads['b1'] = db1.flatten()

    return loss, grads


class ExperimentConvNet(object):
  def __init__(self, input_dim=(3, 32, 32),
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):

    self.params = {}
    self.reg = reg
    self.dtype = dtype

    C, H, W = input_dim
    outC, outH, outW = C, H, W

    """CONV1"""
    inC, inH, inW = outC, outH, outW
    num_filter = 32
    filter_size = 3
    stride = 1
    pad = 1
    self.params['W1'] = weight_scale * np.random.randn(num_filter, inC, filter_size, filter_size)
    self.params['b1'] = np.zeros(num_filter)
    outC, outH, outW = num_filter, (inH - filter_size + 2 * pad) / stride + 1, (inW - filter_size + 2 * pad) / stride + 1

    # """BATCH_NORM1"""
    # inC, inH, inW = outC, outH, outW
    # gamma1, beta1 = np.ones(inC), np.zeros(inC)
    # self.params['gamma1'] = gamma1
    # self.params['beta1'] = beta1
    # outC, outH, outW = inC, inH, inW

    # print(outC, outH, outW)
    """RELU1"""
    pass

    """POOL1"""
    inC, inH, inW = outC, outH, outW
    filter_size = 2
    stride = 2
    outC, outH, outW = inC, (inH - filter_size) / stride + 1, (inW - filter_size) / stride + 1

    """CONV2"""
    inC, inH, inW = outC, outH, outW
    num_filter = 32
    filter_size = 3
    stride = 1
    pad = 1
    self.params['W2'] = weight_scale * np.random.randn(num_filter, inC, filter_size, filter_size)
    self.params['b2'] = np.zeros(num_filter)
    outC, outH, outW = num_filter, (inH - filter_size + 2 * pad) / stride + 1, (inW - filter_size + 2 * pad) / stride + 1

    # """BATCH_NORM2"""
    # inC, inH, inW = outC, outH, outW
    # gamma2, beta2 = np.ones(inC), np.zeros(inC)
    # self.params['gamma2'] = gamma2
    # self.params['beta2'] = beta2
    # outC, outH, outW = inC, inH, inW

    """RELU2"""
    pass

    """POOL2"""
    inC, inH, inW = outC, outH, outW
    filter_size = 2
    stride = 2
    outC, outH, outW = inC, (inH - filter_size) / stride + 1, (inW - filter_size) / stride + 1

    """DENSE1"""
    inC, inH, inW = outC, outH, outW
    inN = inC * inH * inW
    outN = hidden_dim
    self.params['W3'] = weight_scale * np.random.randn(inN, outN)
    self.params['b3'] = np.zeros(outN)

    # """BATCH_NORM3"""
    # inN = outN
    # outN = inN
    # gamma3, beta3 = np.array([1]), np.array([0])
    # self.params['gamma3'] = gamma3
    # self.params['beta3'] = beta3

    """RELU3"""
    pass

    """DENSE2"""
    inN = outN
    outN = num_classes
    self.params['W4'] = weight_scale * np.random.randn(inN, outN)
    self.params['b4'] = np.zeros(outN)

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)

  def loss(self, X, y=None):
    mode = 'test' if y is None else 'train'

    W1, b1 = self.params['W1'], self.params['b1']
    # gamma1 = self.params['gamma1']
    # beta1 = self.params['beta1']
    # bn1_param = {'mode': mode}
    conv1_param = {'stride': 1, 'pad': 1}
    pool1_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    W2, b2 = self.params['W2'], self.params['b2']
    # gamma2 = self.params['gamma2']
    # beta2 = self.params['beta2']
    # bn2_param = {'mode': mode}
    conv2_param = {'stride': 1, 'pad': 1}
    pool2_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    W3, b3 = self.params['W3'], self.params['b3']
    # gamma3 = self.params['gamma3']
    # beta3 = self.params['beta3']
    # bn3_param = {'mode': mode}

    W4, b4 = self.params['W4'], self.params['b4']


    conv1, conv1_cache = conv_relu_pool_forward(X, W1, b1,
                                                      conv1_param,
                                                      pool1_param)

    conv2, conv2_cache = conv_relu_pool_forward(conv1, W2, b2,
                                                      conv2_param,
                                                      pool2_param)

    dense3, dense3_cache = affine_relu_forward(conv2, W3, b3)

    dense4, dense4_cache = affine_forward(dense3, W4, b4)

    scores = dense4
    if y is None:
      return scores

    loss, grads = 0, {}

    loss, dE = softmax_loss(scores, y)
    loss += 0.5 * self.reg * (
    np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W3 ** 2) + np.sum(W4 ** 2))

    dx, dW4, db4 = affine_backward(dE, dense4_cache)
    dx, dW3, db3= affine_relu_backward(dx, dense3_cache)

    dx, dW2, db2= conv_relu_pool_backward(dx, conv2_cache)
    dx, dW1, db1= conv_relu_pool_backward(dx, conv1_cache)

    dW4 += self.reg * W4
    dW3 += self.reg * W3
    dW2 += self.reg * W2
    dW1 += self.reg * W1

    db4 = db4.flatten()
    grads['W4'] = dW4
    grads['b4'] = db4

    db3 = db3.flatten()
    # dgamma3 = dgamma3.sum()
    # dbeta3 = dbeta3.sum()
    grads['W3'] = dW3
    grads['b3'] = db3
    # grads['gamma3'] = dgamma3
    # grads['beta3'] = dbeta3

    grads['W2'] = dW2
    db2 = db2.flatten()
    grads['b2'] = db2
    # grads['gamma2'] = dgamma2
    # grads['beta2'] = dbeta2

    grads['W1'] = dW1
    grads['b1'] = db1.flatten()
    # grads['gamma1'] = dgamma1
    # grads['beta1'] = dbeta1

    return loss, grads
if __name__ == '__main__':
  new = ExperimentConvNet()
