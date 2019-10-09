import tensorflow as tf
import numpy as np
tf.enable_eager_execution()

tensor = tf.constant([[1,2,3],[0, 0, 0], [3,4,5], [0, 0,0]])
tensor = tf.contrib.layers.dense_to_sparse(
    tensor,
    eos_token=0,
    outputs_collections=None,
    scope=None
)

#tensorb = tf.constant()
result = tf.sparse.sparse_dense_matmul(tensor, np.matrix([2,4,6]))

print(result)