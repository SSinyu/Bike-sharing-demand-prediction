import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn, layers


def demand_generator(X, y, batch_size, vocab_x):
    # array X shape : (train_size, 12)
    # array y shape : (train_size, 2)
    for i in range(0, len(y)-batch_size, batch_size):
        batch_x_ = X[i:i+batch_size]
        batch_y_ = y[i:i+batch_size]

        batch_x = np.zeros([batch_size, 12])
        for batch_ind, e_batch in enumerate(batch_x_):
            e_data = np.zeros([12])
            for demand_ind, e_demand in enumerate(e_batch):
                e_data[demand_ind] = vocab_x.get(e_demand)
            batch_x[batch_ind] = e_data

        yield batch_x, batch_y_



def demand_eval_generator(X, y, split_size):
    for i in range(0, len(y)-split_size, split_size):
        split_x_ = X[i:i+split_size]
        split_y_ = y[i:i+split_size]

        yield split_x_, split_y_



def build_validation(X, y, vocab_x):
    # array X shape : (eval_size, 12)
    # array y shape : (eval_size, 2)
    eval_x = np.zeros([len(y), 12])
    for ind, e_seq in enumerate(X):
        e_data = np.zeros([12])
        for demand_ind, e_demand in enumerate(e_seq):
            e_data[demand_ind] = vocab_x.get(e_demand)
        eval_x[ind] = e_data

    return eval_x, y



def length(sequences):
    used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
    seq_len = tf.reduce_sum(used, reduction_indices=1)
    return tf.cast(seq_len, tf.int32)

def feedforwardNN(inputs, name, hidden_size=44):
    with tf.variable_scope(name):
        sfnn = tf.contrib.layers.fully_connected(inputs, hidden_size, activation_fn=tf.nn.tanh)
        return sfnn

def BidirectionalLSTMEncoder(inputs, name, hidden_size=50):
    with tf.variable_scope(name):
        GRU_cell_fw = rnn.LSTMCell(hidden_size)
        GRU_cell_bw = rnn.LSTMCell(hidden_size)
        ((fw_outputs, bw_outputs), (_,_)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw, cell_bw=GRU_cell_bw, inputs=inputs, sequence_length=length(inputs), dtype=tf.float32)
        outputs = tf.concat((fw_outputs, bw_outputs), 2)
        return outputs

def AttentionLayer(inputs, name, hidden_size=50):
    with tf.variable_scope(name):
        u_context = tf.Variable(tf.truncated_normal([hidden_size * 2]), name='u_context')
        h = layers.fully_connected(inputs, hidden_size * 2, activation_fn=tf.nn.tanh)
        alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keepdims=True), dim=1)
        attention_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
        return attention_output


def demand_loss(pred, real, alpha_):
    _pred_in = pred[0]
    _pred_out = pred[1]
    _real_in = real[0]
    _real_out = real[1]
    first_term = tf.reduce_sum(tf.abs(_pred_in - _real_in) + tf.abs(_pred_out - _real_out))
    second_term = tf.log((tf.reduce_sum(_pred_in)+1)/(tf.reduce_sum(_real_in)+1)) + tf.log((tf.reduce_sum(_pred_out)+1)/(tf.reduce_sum(_real_out)+1))
    return first_term + alpha_ * second_term
    
