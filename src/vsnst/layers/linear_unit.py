import tensorflow as tf


class LinearUnit(tf.keras.layers.Layer):
    """Linear unit layer that applies a linear transformation to the input."""

    def __init__(
        self, units, activation=None, use_time_distributed=False, use_bias=True
    ):
        """
        Linear unit layer that applies a linear transformation to the input.

        Args:
            units (int): Number of units in the dense layer.
            activation (str): Activation function to use.
            use_time_distributed (bool): Whether to use time distributed layer.
            use_bias (bool): Whether to use bias in the dense layer.
        """
        super(LinearUnit, self).__init__()
        self.linear = tf.keras.layers.Dense(
            units, activation=activation, use_bias=use_bias
        )
        if use_time_distributed:
            self.linear = tf.keras.layers.TimeDistributed(self.linear)

    def call(self, inputs):
        return self.linear(inputs)
