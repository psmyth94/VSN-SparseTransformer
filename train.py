import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from layers import *

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def build_model(d_model, lr, seq_len, num_features, num_cat = 0, dropout_rate = 0.1, num_classes = 1, use_time_distributed = True,
               total_iterations = 10000, cycle_length = 7000, div_factor = 10., epoch_decay = 1.0):
    """
    input should be (batch_size, seq_len, num_features) and categorical data should be placed first in last dimenstion
    that is: input[:,:,:num_cat] contains categorical variables and input[:,:,num_cat:] contains continuous
    total_iterations: the total number of steps before the cycle repeats.
    cycle_length: the number of steps before the lr starts to decay exponentially
    div_factor: the range of lr is lr/div_factor to lr.
    epoch_decay: after a cycle, lr will decay by lr*epoch_decay**(total_iterations//current_step). If epoch == 1., no decay occurs.
    """
    tf.keras.layers.Input(shape=(seq_len, num_features))
    
    # INIT initializers
    embed_dim = head_dim*num_heads
    initializer_emb = 'uniform'
    initializer = 'glorot_uniform'
    gain = 1.
    

    inp = tf.keras.layers.Input(shape=(seq_len, num_feat))
    
    # CATEGORICAL EMBEDDING
    cat_embeddings = [
        tf.keras.Sequential([
            tf.keras.layers.InputLayer([seq_len]),
            tf.keras.layers.Embedding(
                10,
                d_model,
                input_length=13,
                dtype=tf.float32)
        ])
        for i in range(num_cat)
    ]
    # CONTINUOUS EMBEDDING
    cont_embeddings = [
        LinearUnit(d_model, use_time_distributed=use_time_distributed)
        for i in range(num_cat, num_feat)
    ]
    
    # INPUT EMBEDDING LAYER
    embedded_inputs = [
        cat_embeddings[i](inp[...,i]) if i < num_cat else cont_embeddings[i-num_cat](inp[..., i:i + 1])
        for i in range(num_feat)
    ]
    
    x = K.stack(embedded_inputs, axis=-1) # (batch_size, seq_len, d_model, num_features)
        
    x = VariableSelection(num_feat, d_model, dropout_rate=dropout_rate, 
                      use_time_distributed = use_time_distributed,
                      reduce_sum = True)(x) # (batch_size, seq_len, d_model)

    pos_enc = positional_encoding(seq_len, d_model)
    skip = x
    x = SparseAttention()(x[:,-1:,:], x, x)      
    x,_ = GatedLinearUnit(d_model, dropout_rate=dropout_rate, use_time_distributed=use_time_distributed)(x)
    x = tf.keras.layers.LayerNormalization()(x+skip)
    
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    # x = GatedResidualNetwork(d_model,
    #                          dropout_rate=dropout_rate,
    #                          use_time_distributed=False)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inp, outputs=outputs)
    opt = CLRAdam(learning_rate = lr, cycle_length = cycle_length,
                  total_iterations = total_iterations, div_factor = div_factor,
                  epoch_decay = epoch_decay)
    if num_classes == 1:
        loss = tf.keras.losses.BinaryCrossentropy()
    else:
        loss = tf.keras.losses.CategoricalCrossentropy()
    model.compile(loss=loss, optimizer = opt, sample_weight_mode = "temporal")

    return model
    
