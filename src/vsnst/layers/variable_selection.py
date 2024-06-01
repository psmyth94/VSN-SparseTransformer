import tensorflow as tf
from tensorflow.keras import backend as K

from .gated_residual_network import GatedResidualNetwork


class VariableSelection(tf.keras.layers.Layer):
    def __init__(
        self,
        num_features,
        units,
        dropout_rate,
        use_time_distributed=True,
        reduce_sum=True,
    ):
        """
        Variable selection layer that selects the most relevant features from the input
        based on the attention mechanism described in the paper.

        Args:
            num_features (int): Number of features in the input.
            units (int): Number of units in the GRN.
            dropout_rate (float): Dropout rate.
            use_time_distributed (bool): Whether to use time distributed layer.
            reduce_sum (bool): Whether to reduce the output dimensionality from 4D to 3D.
        """
        super(VariableSelection, self).__init__()
        self.units = units
        self.grns = list()
        # Create a GRN for each feature independently
        self.num_features = num_features
        for idx in range(num_features):
            grn = GatedResidualNetwork(
                units,
                dropout_rate=dropout_rate,
                use_time_distributed=use_time_distributed,
            )
            self.grns.append(grn)

        # Create a GRN for the concatenation of all the features
        self.grn_concat = GatedResidualNetwork(
            units,
            output_size=num_features,
            dropout_rate=dropout_rate,
            use_time_distributed=use_time_distributed,
        )
        self.use_time_distributed = use_time_distributed
        self.softmax = tf.keras.layers.Activation("softmax")
        self.reduce_sum = reduce_sum
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs):
        """
        Selects the most relevant features from the input based on the attention mechanism
        described in the paper.
        """

        dims = inputs.get_shape()
        if len(dims) == 4:
            flatten = K.reshape(inputs, [-1, dims[1], dims[2] * dims[3]])
        elif len(dims) == 3:
            flatten = self.flatten(inputs)
            inputs = tf.expand_dims(inputs, axis=-2)
        else:
            raise ValueError("dimension of inputs must be 3 or 4")

        v = self.grn_concat(flatten)

        # (batch_size, seq_len, 1, num_features) makes v broadcastable
        v = tf.expand_dims(self.softmax(v), axis=-2)

        x = []
        for idx in range(self.num_features):
            x.append(self.grns[idx](inputs[Ellipsis, idx]))
        x = tf.stack(x, axis=-1)
        outputs = tf.keras.layers.Multiply()([v, x])
        # Reduces 4D outputs to 3D
        if self.reduce_sum:
            outputs = K.sum(outputs, axis=-1)
        return outputs
