from gcn.inits import *
import tensorflow as tf
from utils import *
flags = tf.app.flags
FLAGS = flags.FLAGS
import scipy.sparse as sp
# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')


        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):

            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):    
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class AdaptiveGraphConvolution(Layer):
    """Adaptive graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, adj_mat, features, f_out_dim=200, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(AdaptiveGraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        # self.features = tuple_to_sparse(placeholders['features'])
        self.adj = (adj_mat+sp.eye(adj_mat.shape[0])).tocoo()
        self.num_vertex = self.adj.shape[0]
        self.row = list(self.adj.row)
        self.col = list(self.adj.col)
        self.adj = tf.cast(self.adj.A, 1)
        self.dia_adj = tf.reduce_sum(self.adj, 1)

        self.f_in_dim = features.shape[1]
        self.f_out_dim = f_out_dim
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):

            # for i in range(len(self.support)):
            #     self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
            #                                             name='weights_' + str(i))
            # self.vars['f_weights'] = glorot([self.f_in_dim, self.f_out_dim], name='f_weights')
            # self.vars['f_bias'] = zeros([self.f_out_dim], name='f_bias')
            #
            # if self.bias:
            #     self.vars['bias'] = zeros([output_dim], name='bias')
             # adaptive part for feature input

            self.vars['weights_' ] = glorot([input_dim, output_dim],
                                                        name='weights_')

            self.vars['f_weights'] = glorot([output_dim*2, 1], name='f_weights')
            self.vars['f_bias'] = constants([1],0.5, name='f_bias')

            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')
        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        if not self.featureless:
            pre_sup = dot(x, self.vars['weights_'],
                          sparse=self.sparse_inputs)
        else:
            pre_sup = self.vars['weights_']

        features_vi = tf.gather(pre_sup, self.row)
        features_vj = tf.gather(pre_sup, self.col)
        vivj = tf.concat([features_vi, features_vj],1)
        vivj= dot(vivj, self.vars['f_weights'])+self.vars['f_bias']
        # vivj = tf.nn.sigmoid(vivj)

        deg_vi = tf.gather(self.dia_adj, self.row)
        deg_vj = tf.gather(self.dia_adj, self.col)
        deg_vivj = tf.expand_dims(tf.math.multiply(deg_vi, deg_vj),-1)
        deg_vivj_normalized = tf.math.pow(deg_vivj,-vivj)


        # def adapt_preprocess_adj(adj, features):
        #     adj_sl = adj + sp.eye(adj.shape[0])
        #     adp_adj = adj_sl.multiply(features * features.T)
        #
        #     adj = sp.coo_matrix(adp_adj)
        #     rowsum = np.array(adj.sum(1))
        #     d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        #     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        #     d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        #
        #     adj_normalized = adj_sl.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
        #     return sparse_to_tuple(adj_normalized)
        # convolve

        # features = self.features*self.vars['f_weights']+self.vars['f_bias']
        # features = dot(self.features.A,self.vars['f_weights'])+self.vars['f_bias']
        # adj_sl = self.adj + sp.eye(self.adj.shape[0])

        # chebyshev is unable to use.
        # self.support = adapt_preprocess_adj(self.adj)

        support = tf.sparse.SparseTensor(indices=np.stack([self.row,self.col],1),values=tf.squeeze(deg_vivj_normalized),dense_shape=[self.num_vertex,self.num_vertex])


        support = dot(support, pre_sup,sparse=True)
        output = tf.add_n([support])



        # supports = list()
        # for i in range(len(self.support)):
        #     if not self.featureless:
        #         pre_sup = dot(x, self.vars['weights_' + str(i)],
        #                       sparse=self.sparse_inputs)
        #     else:
        #         pre_sup = self.vars['weights_' + str(i)]
        #     support = dot(self.support[i], pre_sup, sparse=True)
        #     supports.append(support)
        # output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)
