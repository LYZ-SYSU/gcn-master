from gcn.inits import *
import tensorflow as tf
from utils import *
flags = tf.app.flags
FLAGS = flags.FLAGS
import scipy.sparse as sp
# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}
"""
From pull request:
https://github.com/tensorflow/tensorflow/pull/21276
"""

from tensorflow.keras.layers import Wrapper
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.ops import nn_impl
from tensorflow.python.keras import initializers

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.eager import context


class WeightNorm(Wrapper):
    """ This wrapper reparameterizes a layer by decoupling the weight's
    magnitude and direction. This speeds up convergence by improving the
    conditioning of the optimization problem.

    Weight Normalization: A Simple Reparameterization to Accelerate
    Training of Deep Neural Networks: https://arxiv.org/abs/1602.07868
    Tim Salimans, Diederik P. Kingma (2016)

    WeightNorm wrapper works for keras and tf layers.

    ```python
      net = WeightNorm(tf.keras.layers.Conv2D(2, 2, activation='relu'),
             input_shape=(32, 32, 3), data_init=True)(x)
      net = WeightNorm(tf.keras.layers.Conv2D(16, 5, activation='relu'),
                       data_init=True)
      net = WeightNorm(tf.keras.layers.Dense(120, activation='relu'),
                       data_init=True)(net)
      net = WeightNorm(tf.keras.layers.Dense(n_classes),
                       data_init=True)(net)
    ```

    Arguments:
      layer: a `Layer` instance.
      data_init: If `True` use data dependent variable initialization

    Raises:
      ValueError: If not initialized with a `Layer` instance.
      ValueError: If `Layer` does not contain a `kernel` of weights
      NotImplementedError: If `data_init` is True and running graph execution
    """
    def __init__(self, layer, data_init=False, **kwargs):
        if not isinstance(layer, Layer):
            raise ValueError(
                'Please initialize `WeightNorm` layer with a '
                '`Layer` instance. You passed: {input}'.format(input=layer))

        if not context.executing_eagerly() and data_init:
            raise NotImplementedError(
                'Data dependent variable initialization is not available for '
                'graph execution')

        self.initialized = True
        if data_init:
            self.initialized = False

        super(WeightNorm, self).__init__(layer, **kwargs)
        self._track_checkpointable(layer, name='layer')

    def _compute_weights(self):
        """Generate weights by combining the direction of weight vector
         with it's norm """
        with variable_scope.variable_scope('compute_weights'):
            self.layer.kernel = nn_impl.l2_normalize(
                self.layer.v, axis=self.norm_axes) * self.layer.g

    def _init_norm(self, weights):
        """Set the norm of the weight vector"""
        from tensorflow.python.ops.linalg_ops import norm
        with variable_scope.variable_scope('init_norm'):
            flat = array_ops.reshape(weights, [-1, self.layer_depth])
            return array_ops.reshape(norm(flat, axis=0), (self.layer_depth,))

    def _data_dep_init(self, inputs):
        """Data dependent initialization for eager execution"""
        from tensorflow.python.ops.nn import moments
        from tensorflow.python.ops.math_ops import sqrt

        with variable_scope.variable_scope('data_dep_init'):
            # Generate data dependent init values
            activation = self.layer.activation
            self.layer.activation = None
            x_init = self.layer.call(inputs)
            m_init, v_init = moments(x_init, self.norm_axes)
            scale_init = 1. / sqrt(v_init + 1e-10)

        # Assign data dependent init values
        self.layer.g = self.layer.g * scale_init
        self.layer.bias = (-m_init * scale_init)
        self.layer.activation = activation
        self.initialized = True

    def build(self, input_shape):
        """Build `Layer`"""
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        self.input_spec = InputSpec(shape=input_shape)

        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = False

            if not hasattr(self.layer, 'kernel'):
                raise ValueError(
                    '`WeightNorm` must wrap a layer that'
                    ' contains a `kernel` for weights'
                )

            # The kernel's filter or unit dimension is -1
            self.layer_depth = int(self.layer.kernel.shape[-1])
            self.norm_axes = list(range(self.layer.kernel.shape.ndims - 1))

            self.layer.v = self.layer.kernel
            self.layer.g = self.layer.add_variable(
                name="g",
                shape=(self.layer_depth,),
                initializer=initializers.get('ones'),
                dtype=self.layer.kernel.dtype,
                trainable=True)

            with ops.control_dependencies([self.layer.g.assign(
                    self._init_norm(self.layer.v))]):
                self._compute_weights()

            self.layer.built = True

        super(WeightNorm, self).build()
        self.built = True

    def call(self, inputs):
        """Call `Layer`"""
        if context.executing_eagerly():
            if not self.initialized:
                self._data_dep_init(inputs)
            self._compute_weights()  # Recompute weights for each forward pass

        output = self.layer.call(inputs)
        return output

    def compute_output_shape(self, input_shape):
        return tensor_shape.TensorShape(
            self.layer.compute_output_shape(input_shape).as_list())

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
        self.fvars = {}  # variables without regularization
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
    def __init__(self, input_dim, output_dim, placeholders, adj_mat, dia_adj, features, f_out_dim=200, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False,power_bias=False, **kwargs):
        super(AdaptiveGraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act

        # self.features = tuple_to_sparse(placeholders['features'])
        self.adj = adj_mat.tocoo()
        self.num_vertex = self.adj.shape[0]
        self.row = list(self.adj.row)
        self.num_relation = len(self.row)
        self.col = list(self.adj.col)
        self.dia_adj = dia_adj

        self.f_in_dim = features.shape[1]
        self.f_out_dim = f_out_dim
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.power_bias = power_bias
        self.constrain_alpha = 0
        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']
        self.scale = tf.Variable(tf.ones([output_dim]))
        self.shift = tf.Variable(tf.zeros([output_dim]))
        self.epsilon = 0.001
        with tf.variable_scope('original'):

            # for i in range(len(self.support)):
            #     self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
            #                                             name='weights_' + str(i))
            # self.vars['f_weights'] = glorot([self.f_in_dim, self.f_out_dim], name='f_weights')
            # self.vars['f_bias'] = zeros([self.f_out_dim], name='f_bias')
            #
            # if self.bias:
            #     self.vars['bias'] = zeros([output_dim], name='bias')
            # adaptive part for feature input

            self.vars['weights_'] = glorot([input_dim, output_dim],
                                                        name='weights_')

            # self.vars['f_bias'] = zeros([1], name='f_bias')

            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        with tf.variable_scope('adp'):
            self.fvars['fs_weights'] = glorot([input_dim, output_dim],
                                             name='fs_weights')

            self.fvars['f_weights'] = glorot([output_dim * 2, 1], name='f_weights')
            self.fvars['f_bias'] = constants([1], 0, name='f_bias')

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
            fpre_sup = dot(x, self.fvars['fs_weights'],
                           sparse=self.sparse_inputs)
        else:
            pre_sup = self.vars['weights_']
            fpre_sup = self.fvars['fs_weights']

        features_vi = tf.gather(fpre_sup, self.row)
        features_vj = tf.gather(fpre_sup, self.col)
        vivj = tf.concat([features_vi, features_vj],1)
        vivj= dot(vivj, self.fvars['f_weights']) + self.fvars['f_bias']
        vivj = tf.nn.sigmoid(vivj)

        pvij = tf.Print(vivj, [vivj],summarize=10,message=self.name)
        if self.power_bias == True:
            pvij = pvij+0.04
        # pvij = vivj
        deg_vi = tf.gather(self.dia_adj, self.row)
        deg_vj = tf.gather(self.dia_adj, self.col)
        deg_vivj = tf.expand_dims(tf.math.multiply(deg_vi, deg_vj),-1)
        pdeg_vivj = deg_vivj
        # pdeg_vivj = tf.Print(deg_vivj, [deg_vivj])
        # t = tf.Print(self.vars['f_bias'],[self.vars['f_bias']])

        # t = tf.nn.sigmoid(self.fvars['f_bias'])/10
        # t = tf.Print(t, [t])
        pdeg_vivj = tf.cast(pdeg_vivj,tf.float32)
        deg_vivj_normalized = tf.math.pow(pdeg_vivj, -pvij)

        self.constrain_alpha = tf.reduce_sum(tf.math.pow(vivj-0.5,2),0)/(self.num_relation)


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

        # batch normalization
        # mean,var = tf.nn.moments(output,axes=[0])
        # pmean = tf.Print(mean,[mean],message=self.name)
        # pvar = tf.Print(var,[var],message=self.name)
        # poutput = tf.Print(output,[output],message=self.name)
        # output = tf.nn.batch_normalization(poutput,mean,var*10,self.scale,self.shift,self.epsilon)
        # poutput2 = tf.Print(output, [output], message=self.name)
        output = self.act(output)
        return output


