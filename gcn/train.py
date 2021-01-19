from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn.utils import *
from gcn.models import GCN, MLP, AdaGCN
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'pubmed', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn_adp', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('f_out_dim', 100, 'Output dimensionality of feature.')

flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')

flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')

flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 50, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

# Some preprocessing
features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
elif FLAGS.model == 'gcn_adp':
    support = [preprocess_adj(adj)]
    # support = [adapt_preprocess_adj(adj, features)]
    num_supports = 1
    model_func = AdaGCN
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))


features_mat = features
features = sparse_to_tuple(features)
f_in_dim = features_mat.shape[1]
adj_mat = adj
adj = sparse_to_tuple(adj)
# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    # 'features_mat': tf.sparse_placeholder(tf.float32, shape=tf.constant(features_mat, dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'f_out_dim': tf.placeholder_with_default(100, shape=()),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}
adj_mat = adj_mat+sp.eye(adj_mat.shape[0])
dia_adj = np.sum(adj_mat.A,1)
dia_adj = tf.convert_to_tensor(dia_adj)
# dia_adj = tf.reduce_sum(tf.cast(adj_mat.A, 1), 1)

# Create model
if model_func == AdaGCN:
    model = model_func(placeholders, adj_mat=adj_mat, dia_adj=dia_adj, features=features_mat, input_dim=features[2][1], logging=True)
else:
    model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


def evaluate_adp(features, support, labels, mask, placeholders,):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)

# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    if FLAGS.model == 'gcn_adp':
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    else:
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    feed_dict.update({placeholders['f_out_dim']: FLAGS.f_out_dim})
    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
    # sess.run()
    # Validation
    if FLAGS.model == 'gcn_adp':
        cost, acc, duration = evaluate_adp(features, support,  y_val, val_mask, placeholders)
    else:
        cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
if FLAGS.model == 'gcn_adp':
    test_cost, test_acc, test_duration = evaluate_adp(features, support, y_test, test_mask, placeholders)
else:
    test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)

print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

