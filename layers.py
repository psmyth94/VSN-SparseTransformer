import tensorflow as tf
import tensorflow.keras.backend as K

class CLRAdam(tf.keras.optimizers.Adam):
    def __init__(self, learning_rate, cycle_length, total_iterations = None, div_factor = 10., epoch_decay = 0.5):
        self.one_cycle_schedule = CLRSchedule(cycle_length, total_iterations, div_factor, epoch_decay)
        lr = lambda: learning_rate * self.one_cycle_schedule(self.iterations)
        momentum = lambda: self.one_cycle_schedule.get_momentum(self.iterations)

        super(CLRAdam, self).__init__(learning_rate=lr, beta_1=momentum, beta_2=0.99)


class CLRSchedule(tf.optimizers.schedules.LearningRateSchedule):
    def __init__(self, cycle_length, total_iterations = None, div_factor = 10., epoch_decay = 0.5):
        self._warmup_end_step = tf.cast(cycle_length * 0.1, dtype=tf.int64)
        self._max_end_step = self._warmup_end_step + tf.cast(cycle_length * 0.4, dtype=tf.int64)
        self._decay_end_step = self._max_end_step + tf.cast(cycle_length * 0.5, dtype=tf.int64)
        self.div_factor = tf.cast(div_factor, dtype = tf.float32)
        self.total_iterations = total_iterations
        self.epoch_decay = epoch_decay
        if total_iterations is not None:
            self.total_iterations = tf.cast(total_iterations, dtype=tf.int64)


    def __call__(self, step):
        epoch = None
        max_factor = 1.
        min_factor = 1/self.div_factor
        if self.total_iterations is not None:
            epoch = (step-1)//self.total_iterations
            min_factor = min_factor*(self.epoch_decay**tf.cast(epoch, dtype = tf.float32))
            max_factor = min_factor*self.div_factor
            step = step - self.total_iterations*epoch
        def warmup():
            # interpolate between initial and max lr
            lr_factor = min_factor + (max_factor-min_factor) * tf.cast(step / self._warmup_end_step, dtype=tf.float32)
            return tf.cast(lr_factor, dtype = tf.float32)

        def max_lr():
            # remain at max for a period
            return tf.cast(max_factor, dtype = tf.float32)

        def initial_decay():
            # decay at half the speed we warmed up at
            decay_step = step - self._max_end_step
            lr_factor = max_factor - (max_factor-min_factor) * tf.cast(decay_step / (self._decay_end_step - self._max_end_step), dtype=tf.float32)
            return tf.cast(lr_factor, dtype = tf.float32)

        def final_decay():
            # then exponential decay from there
            final_step = tf.cast(step - self._decay_end_step, dtype = tf.float32)
            base = tf.cast(1 - 1 / self._decay_end_step, dtype = tf.float32)
            lr_factor = min_factor * tf.math.pow(base, final_step)
            return lr_factor
        
        def default_lr():
            return tf.cast(1/self.div_factor, dtype = tf.float32)

        learning_rate = tf.case([(tf.less_equal(step, self._warmup_end_step), warmup),
                                 (tf.less_equal(step, self._max_end_step), max_lr),
                                 (tf.less_equal(step, self._decay_end_step), initial_decay)],
                                 #(tf.less_equal(step, self.total_iterations), final_decay)],
                                default=final_decay)

        return learning_rate

    def get_momentum(self, step):
        if self.total_iterations is not None:
            step = step - self.total_iterations*((step-1)//self.total_iterations)
        def warmup():
            # interpolate between initial and max momentum
            momentum = 0.95 - 0.1 * (step / self._warmup_end_step)
            return tf.cast(momentum, dtype = tf.float32)

        def max_lr():
            # remain at min momentum while learning rate is maxed
            return tf.cast(0.85, dtype = tf.float32)

        def initial_decay():
            # decay back up to max momentum
            decay_step = step - self._max_end_step
            momentum = 0.85 + 0.1 * (decay_step / (self._decay_end_step - self._max_end_step))
            return tf.cast(momentum, dtype = tf.float32)

        def final_decay():
            # remain at highest momentum
            return tf.cast(0.95, dtype = tf.float32)

        learning_rate = tf.case([(tf.less_equal(step, self._warmup_end_step), warmup),
                                 (tf.less_equal(step, self._max_end_step), max_lr),
                                 (tf.less_equal(step, self._decay_end_step), initial_decay)],
                                default=final_decay)

        return learning_rate

class LinearUnit(tf.keras.layers.Layer):
    def __init__(self, units, activation = None, use_time_distributed = False, use_bias=True):
        super(LinearUnit, self).__init__()
        self.linear = tf.keras.layers.Dense(units, activation=activation, use_bias=use_bias)
        if use_time_distributed:
            self.linear = tf.keras.layers.TimeDistributed(self.linear)

    def call(self, inputs):
        return self.linear(inputs)

class GatedLinearUnit(tf.keras.layers.Layer):
    def __init__(self, units, dropout_rate = None, use_time_distributed=True, activation = None):
        super(GatedLinearUnit, self).__init__()
        self.dropout_rate = dropout_rate
        if dropout_rate is not None:
            self.dropout = tf.keras.layers.Dropout(dropout_rate)

        if use_time_distributed:
            self.activation_layer = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(units, activation=activation)
            )
            self.gated_layer = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(units, activation='sigmoid')
            )
        else:
            self.activation_layer = tf.keras.layers.Dense(units, activation=activation)
            self.gated_layer = tf.keras.layers.Dense(units, activation='sigmoid')

    def call(self, x, training, return_gate = True):
        if self.dropout_rate is not None:
            x = self.dropout(x, training=training)
        gated_out = self.gated_layer(x)
        if return_gate:
            return tf.keras.layers.Multiply()([self.activation_layer(x), gated_out]), gated_out
        else:
            return tf.keras.layers.Multiply()([self.activation_layer(x), gated_out])
class GatedResidualNetwork(tf.keras.layers.Layer):
    def __init__(self, units, output_size = None, dropout_rate = None,
                 use_time_distributed = True, normalize = True):
        super(GatedResidualNetwork, self).__init__()
        
        self.output_size = output_size
        if self.output_size is None:
            self.glu = GatedLinearUnit(
                units,
                dropout_rate=dropout_rate,
                use_time_distributed=use_time_distributed,
                activation=None
            )
        else:
            self.linear = tf.keras.layers.Dense(output_size)
            if use_time_distributed:
                self.linear = tf.keras.layers.TimeDistributed(self.linear)
            self.glu = GatedLinearUnit(
                output_size,
                dropout_rate=dropout_rate,
                use_time_distributed=use_time_distributed,
                activation=None
            )
            
        self.hidden = LinearUnit(
            units,
            activation=None,
            use_time_distributed=use_time_distributed
        )
        self.hidden_additional_context = LinearUnit(
            units,
            activation=None,
            use_time_distributed=use_time_distributed,
            use_bias=False
        )
        self.hidden2 = LinearUnit(
            units,
            activation=None,
            use_time_distributed=use_time_distributed
        )

        self.activation = tf.keras.layers.Activation('elu')
        self.normalize = normalize
        if normalize:
            self.norm = tf.keras.layers.LayerNormalization()

    def call(self, x, additional_context=None, return_gate=False):

        # Setup skip connection
        if self.output_size is None:
            skip = x
        else:
            skip = self.linear(x)

        # Apply feedforward network
        
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

class VariableSelection(tf.keras.layers.Layer):
    def __init__(self, num_features, units, dropout_rate, use_time_distributed = True,
                 reduce_sum = True):
        super(VariableSelection, self).__init__()
        self.units = units
        self.grns = list()
        # Create a GRN for each feature independently
        self.num_features = num_features
        for idx in range(num_features):
            grn = GatedResidualNetwork(units,
                                       dropout_rate=dropout_rate,
                                       use_time_distributed=use_time_distributed)
            self.grns.append(grn)
        
        # Create a GRN for the concatenation of all the features
        self.grn_concat = GatedResidualNetwork(units,
                                               output_size=num_features,
                                               dropout_rate=dropout_rate,
                                               use_time_distributed=use_time_distributed)
        self.use_time_distributed = use_time_distributed
        self.softmax = tf.keras.layers.Activation('softmax')
        self.reduce_sum = reduce_sum
        self.flatten = tf.keras.layers.Flatten()
    def call(self, inputs):
        dims = inputs.get_shape()
        if len(dims) == 4:
            flatten = K.reshape(inputs, [-1, dims[1], dims[2] * dims[3]])
        elif len(dims) == 3:
            flatten = self.flatten(inputs)
            inputs = tf.expand_dims(inputs, axis = -2)
        else:
            raise ValueError("dimension of inputs must be 3 or 4")
        
        v = self.grn_concat(flatten)
        
        # (batch_size, seq_len, 1, num_features) makes v broadcastable
        v = tf.expand_dims(self.softmax(v), axis = -2)

        x = []
        for idx in range(self.num_features):
            x.append(self.grns[idx](inputs[Ellipsis,idx]))
        x = tf.stack(x, axis=-1)
        outputs = tf.keras.layers.Multiply()([v, x])
        # Reduces 4D outputs to 3D
        if self.reduce_sum:
            outputs = K.sum(outputs, axis=-1)
        return outputs

      
def entmax(z, alpha, n_iter=100):
    """
    Taken from https://github.com/flaviagiammarino/tcan-tensorflow. Refer to this github page for more details.
    """
    if alpha < 1:
        raise ValueError('The sparsity parameter should be greater than or equal to 1.')

    elif alpha == 1:
        # Calculate the softmax probabilities.
        return tf.nn.softmax(tf.cast(z, tf.float32), axis=-1)

    else:
        # Calculate the entmax probabilities.
        z = (alpha - 1) * tf.cast(z, tf.float32)
        z_max = tf.reduce_max(z, axis=-1, keepdims=True)
        tau_min = z_max - 1
        tau_max = z_max - (z.shape[-1]) ** (1 - alpha)

        for _ in tf.range(n_iter):
            tau = (tau_min + tau_max) / 2
            p = tf.maximum(z - tau, 0.0) ** (1 / (alpha - 1))
            Z = tf.reduce_sum(p, axis=-1, keepdims=True)
            tau_min = tf.where(Z >= 1, tau, tau_min)
            tau_max = tf.where(Z < 1, tau, tau_max)

        return p / Z

class SparseAttention(tf.keras.layers.Layer):
    def __init__(self, alpha=1.5):
        """
        Taken from https://github.com/flaviagiammarino/tcan-tensorflow. Refer to this github page for more details.
        """
        self.alpha = alpha
        super(SparseAttention, self).__init__()

    def call(self, query, key, value = None, return_attention_scores=False):

        # Extract the query, value and key matrices.
        if value is None:
            value = key

        # Calculate the attention scores.
        scores = tf.matmul(query, key, transpose_b=True)

        # Calculate the attention weights.
        weights = entmax(scores, alpha=self.alpha)

        # Calculate the context vector.
        outputs = tf.matmul(weights, value)

        if return_attention_scores:
            return outputs, weights

        else:
            return outputs
