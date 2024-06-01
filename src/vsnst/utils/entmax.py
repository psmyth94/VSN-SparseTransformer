import tensorflow as tf


def sparsemax(z):
    """Creates a sparse probability distribution from the input tensor z."""
    z = tf.cast(z, tf.float32)
    z_sorted, _ = tf.nn.top_k(z, k=z.shape[-1])
    z_cumsum = tf.cumsum(z_sorted, axis=-1)
    k = tf.range(1, z.shape[-1] + 1, dtype=tf.float32)
    z_check = 1 + k * z_sorted > z_cumsum
    k_z = tf.reduce_sum(tf.cast(z_check, tf.float32), axis=-1)
    tau = (tf.reduce_sum(z_sorted, axis=-1) - 1) / k_z
    return tf.maximum(z - tau[:, None], 0.0)


def entmax(z, alpha, n_iter=100):
    """
    The entmax activation function is a generalization of the softmax and the sparsemax activation functions.
    It is controlled by the alpha parameter, which can be set to 1 for softmax, 2 for sparsemax, and any value
    greater than 2 for entmax.

    Taken from https://github.com/flaviagiammarino/tcan-tensorflow. Refer to this github page for more details.
    """
    if alpha < 1:
        raise ValueError("The sparsity parameter should be greater than or equal to 1.")

    elif alpha == 1:
        return tf.nn.softmax(tf.cast(z, tf.float32), axis=-1)
    elif alpha == 2:
        sparsemax(z)
    else:
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
