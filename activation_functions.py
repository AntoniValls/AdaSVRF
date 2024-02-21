#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def ReLU(x):
    '''ReLU activation function'''
    return np.maximum(0,x)

def dReLU(x):
    '''Derivative of ReLU activation function'''
    return 1 * (x > 0)

def softmax(z):
    '''Softmax function for multi-class classification output'''
    z = z - np.max(z, axis=1, keepdims=True)
    z_out = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    return z_out

