# -*- coding: utf-8 -*-
import math

from scipy.linalg import hadamard

import tensorflow as tf


def notdense(inputs, units,
             activation=None,
             normalize_inputs=False,
             use_bias=False,
             alpha=None,
             eps=1e-8,
             fixed_q=True,
             fixed_alpha=True,
             trainable=True,
             name='notdense',
             reuse=None):
  """Functional interface for the fixed classifier layer.

  This layer implements the operation: outputs = activation(inputs * H^T + bias)

  Arguments:
    inputs: Tensor input.
    units: Integer or Long, dimensionality of the output space.
    alpha: (optional) scaling factor, defaults to `1 / sqrt(units)`.
    eps: constant error bias.
    fixed_q: use a constant hadamard matrix.
    fixed_alpha: use a constant alpha value.
    activation: Activation function (callable). Set it to None to maintain a linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    trainable: Boolean, if True also add variables to the graph collection
               GraphKeys.TRAINABLE_VARIABLES (see tf.Variable). This option is overriden for the
               variables `q` and `alpha` if `fixed_q` and `fixed_alpha` are set respectively.
    name: String, the name of the layer.
    reuse: Boolean, whether to reuse the weights of a previous layer by the same name.

  Reference:
    Hoffer et al., https://arxiv.org/pdf/1801.04540.pdf
  """
  n_inputs = inputs.get_shape().as_list()[-1]

  # ensure N is a power of two for hadamard construction
  N = 2 ** int(math.ceil(math.log(max(n_inputs, units), 2)))
  hm = hadamard(N)

  # initial scaling factor
  init_alpha = 1. / math.sqrt(units)

  with tf.variable_scope(name, reuse=reuse):
    # normalize x {-1, 1}
    if normalize_inputs:
      inputs = inputs / (tf.norm(inputs, 2, axis=-1, keep_dims=True) + eps)

    # fixed or trainable hadamard matrix
    q = tf.stop_gradient(
        tf.get_variable(
            'q', hm.shape,
            dtype=inputs.dtype,
            trainable=trainable and not fixed_q,
            initializer=tf.constant_initializer(hm)))

    # softmax scalar, inverse of softmax temperature scaling.
    alpha = tf.stop_gradient(
        tf.get_variable(
            'alpha', (1,),
            dtype=inputs.dtype,
            trainable=trainable and not fixed_alpha,
            initializer=tf.constant_initializer(
                init_alpha if alpha is None else alpha)))

    # truncated, orthogonal hadamard
    # TODO, this can be optimized further with the use of a binarized sign + add.
    y = alpha * tf.matmul(inputs, tf.transpose(q[:units, :n_inputs]))

    if use_bias:
      bias = tf.random_uniform(
          (units,),
          minval=-init_alpha,
          maxval=init_alpha,
          dtype=inputs.dtype,
          name='b')
      y += tf.expand_dims(bias, 0)

    if activation is not None:
      y = activation(y)
    return y
