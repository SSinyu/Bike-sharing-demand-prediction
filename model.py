import tensorflow as tf
from tensorflow.contrib import rnn, layers

class RNN_demand:
    def __init__(self, vocab_size, n_predicts, input_length, embedding_size, hidden_size=200):
        self.vocab_size = vocab_size
        self.n_predicts = n_predicts
        self.input_length = input_length
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        with tf.name_scope('placeholder'):
            self.input_x = tf.placeholder(tf.int32, [None, self.input_length], name='input_x')
            self.input_y = tf.placeholder(tf.float32, [None, self.n_predicts], name='input_y')
            self.embed_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.embedding_size])

        with tf.device("/cpu:0"):
            self.embed = tf.Variable(tf.constant(0.0, shape=[self.vocab_size, self.embedding_size]), trainable=False, name='embed')

        with tf.name_scope('time_embedding'):
            self.embed_init = self.embed.assign(self.embed_placeholder)
            time_vec = tf.nn.embedding_lookup(self.embed, self.input_x)

        demand_encode, demand_alpha = self.SFNN_BiLSTM_Attn(time_vec)
        out = self.Encode_prediction(demand_encode)
        self.demand_alpha = demand_alpha
        self.out = out
        #loss = self.model_loss(out, self.input_y)
        #self.loss = loss

    def SFNN_BiLSTM_Attn(self, inputs):
        demand_encode = self.feedforwardNN(inputs, name='SFNN', hidden_size=self.embedding_size)
        demand_encode = self.BidirectionalLSTMEncoder(demand_encode, name='BiLSTM')
        demand_encode, demand_alpha = self.AttentionLayer(demand_encode, name='Attn')
        return demand_encode, demand_alpha

    def Encode_prediction(self, inputs):
        out = layers.fully_connected(inputs, 100, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, self.n_predicts, activation_fn=tf.nn.relu)
        return out
    '''
    def model_loss(self, out, real_out):
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.square(out - real_out), name='loss_mse')
            return loss
    '''
    def feedforwardNN(self, inputs, name, hidden_size):
        with tf.variable_scope(name):
            sfnn = tf.contrib.layers.fully_connected(inputs, hidden_size, activation_fn=tf.nn.tanh)
            return sfnn

    def BidirectionalLSTMEncoder(self, inputs, name):
        with tf.variable_scope(name):
            LSTM_cell_fw = rnn.LSTMCell(self.hidden_size)
            LSTM_cell_bw = rnn.LSTMCell(self.hidden_size)
            ((fw_outputs, bw_outputs), (_,_)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=LSTM_cell_fw, cell_bw=LSTM_cell_bw, inputs=inputs, sequence_length=self.length(inputs), dtype=tf.float32)
            outputs = tf.concat((fw_outputs, bw_outputs), 2)
            return outputs

    def AttentionLayer(self, inputs, name):
        with tf.variable_scope(name):
            u_context = tf.Variable(tf.truncated_normal([self.hidden_size * 2]), name='u_context')
            h = layers.fully_connected(inputs, self.hidden_size*2, activation_fn=tf.nn.tanh)
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keepdims=True), dim=1)
            attention_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
            return attention_output, alpha

    def length(self, sequences):
        used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
        seq_len = tf.reduce_sum(used, reduction_indices=1)
        return tf.cast(seq_len, tf.int32)






def AutoE(input_, input_feature, n_hidden, bias=False):
    # n_hidden type -> list, ex) [300,300,300]
    num_layers = len(n_hidden)

    if bias == True:
        Wh_first = tf.Variable(tf.truncated_normal([input_feature, n_hidden[0]]))
        bh_first = tf.Variable(tf.truncated_normal([n_hidden[0]]))
        layer_x = tf.nn.dropout(tf.nn.relu(tf.matmul(input_, Wh_first) + bh_first), keep_prob=.5)
        if num_layers >= 3:
            for hidden_i in range(num_layers-2):
                Wh_x = tf.Variable(tf.truncated_normal([n_hidden[hidden_i], n_hidden[hidden_i+1]]))
                bh_x = tf.Variable(tf.truncated_normal([n_hidden[hidden_i+1]]))
                layer_x = tf.nn.dropout(tf.nn.relu(tf.matmul(layer_x, Wh_x) + bh_x), keep_prob=.5)
        Wh_last = tf.Variable(tf.truncated_normal([n_hidden[-1], input_feature]))
        bh_last = tf.Variable(tf.truncated_normal([input_feature]))
        out = tf.nn.relu(tf.matmul(layer_x, Wh_last) + bh_last)
        return out, layer_x

    else:
        Wh_first = tf.Variable(tf.truncated_normal([input_feature, n_hidden[0]]))
        layer_x = tf.nn.dropout(tf.nn.relu(tf.matmul(input_, Wh_first)), keep_prob=.5)
        if num_layers >= 3:
            for hidden_i in range(num_layers-2):
                Wh_x = tf.Variable(tf.truncated_normal([n_hidden[hidden_i], n_hidden[hidden_i+1]]))
                layer_x = tf.nn.dropout(tf.nn.relu(tf.matmul(layer_x, Wh_x)), keep_prob=.5)
        Wh_last = tf.Variable(tf.truncated_normal([n_hidden[-1], input_feature]))
        out = tf.nn.relu(tf.matmul(layer_x, Wh_last))
        return out, layer_x

def predict_demand(last_hidden_output, last_hidden, output_feature):
    W = tf.Variable(tf.truncated_normal([last_hidden, output_feature]))
    b = tf.Variable(tf.truncated_normal([output_feature]))
    y_pred = tf.nn.relu(tf.matmul(last_hidden_output, W) + b)
    return y_pred






class GraphConvLayer:
    def __init__(self, in_feature, out_feature, bias=False):
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.bias = bias
        self.W_fc = tf.Variable(tf.truncated_normal(shape=[self.in_feature, self.out_feature]))
        self.b_fc = tf.Variable(tf.constant(0.01, shape=[out_feature]))

    def fw(self, input_, adj):
        if self.bias == True:
            out = tf.matmul(input_, self.W_fc) + self.b_fc
        else:
            out = tf.matmul(input_, self.W_fc)
        out = tf.matmul(adj, out)
        return out

class GCN_demand:
    def __init__(self, n_station, n_feature, n_hidden, n_predict, dropout_):
        self.n_station = n_station
        self.n_feature = n_feature
        self.n_hidden = n_hidden
        self.n_predict = n_predict
        self.dropout_ = dropout_
        self.graphconv1 = GraphConvLayer(self.n_feature, self.n_hidden)
        self.graphconv2 = GraphConvLayer(self.n_hidden, self.n_hidden)
        self.graphconv3 = GraphConvLayer(self.n_hidden, self.n_predict)

        with tf.name_scope("placeholder"):
            self.input_x = tf.placeholder(tf.float32, [self.n_station, self.n_feature])
            self.input_y = tf.placeholder(tf.float32, [self.n_station, 1])

        with tf.device("/cpu:0"):
            self.adj_matrix = tf.Variable(tf.truncated_normal([self.n_station, self.n_station]), trainable=True, name="adj_matrix")

        graph1 = tf.nn.dropout(self.graphconv1.fw(self.input_x, self.adj_matrix), self.dropout_)
        graph2 = tf.nn.dropout(self.graphconv2.fw(graph1, self.adj_matrix), self.dropout_)
        graph3 = self.graphconv3.fw(graph2, self.adj_matrix)
        self.graph3 = graph3
        
