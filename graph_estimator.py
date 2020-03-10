from __future__ import absolute_import, division, print_function, unicode_literals

#from graph_nets import blocks
#from graph_nets import graphs
from graph_nets import modules
#from graph_nets import utils_np
from graph_nets import utils_tf

import numpy as np
import sonnet as snt
import tensorflow as tf

# tf.logging.set_verbosity(tf.logging.INFO)
# tf.reset_default_graph()

GLOBAL_SIZE = 4
NODE_SIZE = 5
EDGE_SIZE = 6
num_nodes = 10
num_edges = 10

OUTPUT_GLOBAL_SIZE = 4
OUTPUT_NODE_SIZE = 5
OUTPUT_EDGE_SIZE = 6

final_output_size = 5

def graph_model_fn(features, labels, mode):
    # What I really need to experiment here is the ability to create the template graph,
    # manipulate the template graph based on the input features, pass it through a graph module,
    # extract the global state, and use the global state for inference.

    # The template graph
    globals = np.zeros([GLOBAL_SIZE]).astype(np.float32)
    nodes = np.zeros([num_nodes, NODE_SIZE]).astype(np.float32)
    edges = np.zeros([num_edges, EDGE_SIZE]).astype(np.float32)
    senders = np.random.randint(num_nodes, size=num_edges, dtype=np.int32)
    receivers = np.random.randint(num_nodes, size=num_edges, dtype=np.int32)

    graph_dict = {"globals": globals,
          "nodes": nodes,
          "edges": edges,
          "senders": senders,
          "receivers": receivers}

    """Model function for CNN."""
    # Input Layer
    input = features["x"]

    batch_of_tensor_data_dicts = [graph_dict for i in range(32)]  # This assumes you have a fixed batch size.
    batch_of_graphs = utils_tf.data_dicts_to_graphs_tuple(batch_of_tensor_data_dicts)
    batch_of_graphs = batch_of_graphs.replace(nodes=tf.reshape(input, [-1, NODE_SIZE]))

    graph_network = modules.GraphNetwork(
        edge_model_fn=lambda: snt.Linear(output_size=OUTPUT_EDGE_SIZE),
        node_model_fn=lambda: snt.Linear(output_size=OUTPUT_NODE_SIZE),
        global_model_fn=lambda: snt.Linear(output_size=OUTPUT_GLOBAL_SIZE))

    output = graph_network(batch_of_graphs)

    output_global = output.globals
    dense_layer = tf.keras.layers.Dense(final_output_size)
    logits = dense_layer(output_global)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        # "probabilities": tf.contrib.sparsemax.sparsemax(logits, name="softmax_tensor")
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

train_data = np.random.randn(60000, num_nodes, NODE_SIZE).astype(np.float32)
train_labels = np.random.choice(final_output_size, 60000)

eval_data = np.random.randn(64, num_nodes, NODE_SIZE).astype(np.float32)
eval_labels = np.random.choice(final_output_size, 64)

# Create the Estimator
mirrored_strategy = tf.distribute.MirroredStrategy()
config = tf.estimator.RunConfig(
    train_distribute=mirrored_strategy, eval_distribute=mirrored_strategy)
mnist_classifier = tf.estimator.Estimator(model_fn=graph_model_fn, model_dir="./model", config=config)

# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}

# logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

def train_input_fn():
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(({'x': train_data}, train_labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(32)

    # Return the dataset.
    return dataset

def eval_input_fn():
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(({'x': eval_data}, eval_labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).batch(32)

    # Return the dataset.
    return dataset

# train one step and display the probabilties
mnist_classifier.train(
    input_fn=train_input_fn,
    steps=1)

mnist_classifier.train(input_fn=train_input_fn, steps=10)

print("FINISHED TRAINING")

eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)
