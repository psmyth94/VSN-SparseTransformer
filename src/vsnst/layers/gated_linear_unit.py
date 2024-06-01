import tensorflow as tf


class GatedLinearUnit(tf.keras.layers.Layer):
    def __init__(
        self, units, dropout_rate=None, use_time_distributed=True, activation=None
    ):
        """
        Gated Linear Unit which is a combination of a linear unit and a sigmoid unit.
        Enables the network to learn to scale the input activations. This is useful
        for learning to ignore certain features.

        Args:
            units (int): Number of units in the dense layer.
            dropout_rate (float): Dropout rate.
            use_time_distributed (bool): Whether to use time distributed layer.
            activation (str): Activation function to use. If None, no activation is applied.
        """
        super(GatedLinearUnit, self).__init__()
        self.dropout_rate = dropout_rate
        if dropout_rate is not None:
            self.dropout = tf.keras.layers.Dropout(dropout_rate)

        if use_time_distributed:
            self.activation_layer = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(units, activation=activation)
            )
            self.gated_layer = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(units, activation="sigmoid")
            )
        else:
            self.activation_layer = tf.keras.layers.Dense(units, activation=activation)
            self.gated_layer = tf.keras.layers.Dense(units, activation="sigmoid")

    def call(self, x, training, return_gate=True):
        """apply gated linear unit to the input. Use dropout if specified."""
        if self.dropout_rate is not None:
            x = self.dropout(x, training=training)
        gated_out = self.gated_layer(x)
        if return_gate:
            return tf.keras.layers.Multiply()(
                [self.activation_layer(x), gated_out]
            ), gated_out
        else:
            return tf.keras.layers.Multiply()([self.activation_layer(x), gated_out])
