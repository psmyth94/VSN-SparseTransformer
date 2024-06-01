import tensorflow as tf

from .linear_unit import LinearUnit
from .gated_linear_unit import GatedLinearUnit


class GatedResidualNetwork(tf.keras.layers.Layer):
    def __init__(
        self,
        units,
        output_size=None,
        dropout_rate=None,
        use_time_distributed=True,
        normalize=True,
    ):
        """
        Gated Residual Network layer that applies a feedforward network with a gated linear unit
        to the input. The output is the element-wise sum of the input and the gated output.

        Args:
            units (int): Number of units in the dense layer.
            output_size (int): Size of the output layer. If None, no output layer is applied.
            dropout_rate (float): Dropout rate.
            use_time_distributed (bool): Whether to use time distributed layer.
            normalize (bool): Whether to apply layer normalization.
        """

        super(GatedResidualNetwork, self).__init__()

        self.output_size = output_size
        if self.output_size is None:
            self.glu = GatedLinearUnit(
                units,
                dropout_rate=dropout_rate,
                use_time_distributed=use_time_distributed,
                activation=None,
            )
        else:
            self.linear = tf.keras.layers.Dense(output_size)
            if use_time_distributed:
                self.linear = tf.keras.layers.TimeDistributed(self.linear)
            self.glu = GatedLinearUnit(
                output_size,
                dropout_rate=dropout_rate,
                use_time_distributed=use_time_distributed,
                activation=None,
            )

        self.hidden = LinearUnit(
            units, activation=None, use_time_distributed=use_time_distributed
        )
        self.hidden_additional_context = LinearUnit(
            units,
            activation=None,
            use_time_distributed=use_time_distributed,
            use_bias=False,
        )
        self.hidden2 = LinearUnit(
            units, activation=None, use_time_distributed=use_time_distributed
        )

        self.activation = tf.keras.layers.Activation("elu")
        self.normalize = normalize
        if normalize:
            self.norm = tf.keras.layers.LayerNormalization()

    def call(self, x, additional_context=None, return_gate=False):
        """apply gated residual network to the input.

        The additional context is used to condition the output of the feedforward network.

        Args:
            x (tf.Tensor): Input tensor.
            additional_context (tf.Tensor): Additional context tensor.
            return_gate (bool): Whether to return the gate values.
        """
        if self.output_size is None:
            skip = x
        else:
            skip = self.linear(x)


        hidden_out = self.hidden(x)
        if additional_context is not None:
            hidden_out = hidden_out + self.hidden_additional_context(additional_context)
        hidden_out = self.activation(hidden_out)
        hidden_out = self.hidden2(hidden_out)

        gated_out, weights = self.glu(hidden_out)
        if return_gate:
            if self.normalize:
                return self.norm(skip + gated_out), weights
            else:
                return skip + gated_out, weights
        else:
            if self.normalize:
                return self.norm(skip + gated_out)
            else:
                return skip + gated_out
