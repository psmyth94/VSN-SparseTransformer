import tensorflow as tf

from vsnst.utils.entmax import entmax


class SparseAttention(tf.keras.layers.Layer):
    def __init__(self, alpha=1.5):
        """
        Sparse attention mechanism using the entmax activation function.

        The entmax activation function is a generalization of the softmax and
        the sparsemax activation functions. It is controlled by the alpha
        parameter, which can be set to 1 for softmax, 2 for sparsemax, and
        any value greater than 2 for entmax.

        Taken from https://github.com/flaviagiammarino/tcan-tensorflow. Refer to this
        github page for more details.

        Args:
            alpha (float): Sparsity parameter. Default is 1.5.
        """
        self.alpha = alpha
        super(SparseAttention, self).__init__()

    def call(self, query, key, value=None, return_attention_scores=False):
        """
        compute the attention mechanism using the entmax activation function

        Args:
            query (tf.Tensor): Query matrix.
            key (tf.Tensor): Key matrix.
            value (tf.Tensor): Value matrix.
            return_attention_scores (bool): Whether to return the attention scores.
        """

        if value is None:
            value = key

        scores = tf.matmul(query, key, transpose_b=True)

        weights = entmax(scores, alpha=self.alpha)

        outputs = tf.matmul(weights, value)

        if return_attention_scores:
            return outputs, weights

        else:
            return outputs
