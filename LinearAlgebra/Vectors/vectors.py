import numpy as np
import torch
import tensorflow as tf

# numpy
v1 = np.array([1,2,3])
v2 = np.array([4,5,6])
result = v1 + v2
print("Numpy result  : {}".format(result))

# PyTorch
v1 = torch.tensor([1,2,3])
v2 = torch.tensor([4,5,6])
result = v1 + v2
print("PyTorch result : {}".format(result))

# TensorFlow
v1 = tf.constant([1,2,3])
v2 = tf.constant([4,5,6])
result = v1 + v2
print("TensorFlow result : {}".format(result))