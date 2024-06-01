import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K
from vsnst import (
    CLRAdam,
    GatedLinearUnit,
    LinearUnit,
    SparseAttention,
    VariableSelection,
)


def get_angles(pos, i, d_model):
    """
    calculate the angle rates for the positional encoding

    Args:
        pos (np.array): Should be of shape (seq_len, 1)
        i (np.array): Should be of shape (1, d_model)
        d_model (int): The dimension of the model
    """

    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model
    )
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def build_model(
    d_model,
    lr,
    seq_len,
    num_features,
    num_cat=0,
    num_heads=4,
    num_layers=1,
    num_units=128,
    num_gated_units=128,
    dropout_rate=0.1,
    num_classes=1,
    use_time_distributed=True,
    total_iterations=10000,
    cycle_length=7000,
    div_factor=10.0,
    epoch_decay=1.0,
):
    """
    The model builder function

    Args:
        d_model (int): The dimension of the model
        lr (float): The learning rate
        seq_len (int): The sequence length
        num_features (int): The number of features
        num_cat (int): The number of categorical features
        num_heads (int): The number of attention heads
        num_layers (int): The number of layers
        num_units (int): The number of units
        num_gated_units (int): The number of gated units
        dropout_rate (float): The dropout rate
        num_classes (int): The number of classes
        use_time_distributed (bool): Whether to use time distributed layer
        total_iterations (int): The total number of iterations
        cycle_length (int): The cycle length
        div_factor (float): The division factor
        epoch_decay (float): The epoch decay

    Returns:
        tf.keras.Model: The model
    """
    tf.keras.layers.Input(shape=(seq_len, num_features))

    inp = tf.keras.layers.Input(shape=(seq_len, num_features))

    # CATEGORICAL EMBEDDING
    cat_embeddings = [
        tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer([seq_len]),
                tf.keras.layers.Embedding(
                    10, d_model, input_length=13, dtype=tf.float32
                ),
            ]
        )
        for i in range(num_cat)
    ]
    # CONTINUOUS EMBEDDING
    cont_embeddings = [
        LinearUnit(d_model, use_time_distributed=use_time_distributed)
        for i in range(num_cat, num_features)
    ]

    # INPUT EMBEDDING LAYER
    embedded_inputs = [
        cat_embeddings[i](inp[..., i])
        if i < num_cat
        else cont_embeddings[i - num_cat](inp[..., i : i + 1])
        for i in range(num_features)
    ]

    x = K.stack(
        embedded_inputs, axis=-1
    )  # (batch_size, seq_len, d_model, num_features)

    x = VariableSelection(
        num_features,
        d_model,
        dropout_rate=dropout_rate,
        use_time_distributed=use_time_distributed,
        reduce_sum=True,
    )(x)  # (batch_size, seq_len, d_model)

    skip = x
    x = SparseAttention()(x[:, -1:, :], x, x)
    x, _ = GatedLinearUnit(
        d_model, dropout_rate=dropout_rate, use_time_distributed=use_time_distributed
    )(x)
    x = tf.keras.layers.LayerNormalization()(x + skip)

    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    # didn't end up using this
    # x = GatedResidualNetwork(d_model,
    #                          dropout_rate=dropout_rate,
    #                          use_time_distributed=False)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inp, outputs=outputs)
    opt = CLRAdam(
        learning_rate=lr,
        cycle_length=cycle_length,
        total_iterations=total_iterations,
        div_factor=div_factor,
        epoch_decay=epoch_decay,
    )
    if num_classes == 1:
        loss = tf.keras.losses.BinaryCrossentropy()
    else:
        loss = tf.keras.losses.CategoricalCrossentropy()
    model.compile(loss=loss, optimizer=opt, sample_weight_mode="temporal")

    return model


def create_data():
    # Define parameters
    seq_len = 50
    num_features = 10
    num_samples = 100

    # Generate random data
    X_train = np.random.randint(0, 10, (num_samples, seq_len, num_features)).astype(
        np.float32
    )
    y_train = np.random.randint(0, 2, (num_samples, 1)).astype(np.float32)

    # Save to CSV files
    df_X_train = pd.DataFrame(X_train.reshape(num_samples, -1))
    df_y_train = pd.DataFrame(y_train, columns=["label"])

    # Save to CSV
    df_X_train.to_csv("/mnt/data/X_train.csv", index=False)
    df_y_train.to_csv("/mnt/data/y_train.csv", index=False)

    return X_train.shape, y_train.shape


def test_vsn():
    X_train, y_train = create_data()

    d_model = 32
    lr = 0.001
    seq_len = 50
    num_features = 10
    num_cat = 2
    num_classes = 1

    # Build the model
    model = build_model(
        d_model=d_model,
        lr=lr,
        seq_len=seq_len,
        num_features=num_features,
        num_cat=num_cat,
        num_classes=num_classes,
    )

    # Print model summary
    model.summary()

    # Train the model
    model.fit(X_train, y_train, epochs=5, batch_size=16)
