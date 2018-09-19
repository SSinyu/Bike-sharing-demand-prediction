import os
import pickle
import time
import numpy as np
import pandas as pd
from operator import itemgetter
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from bicycle_util import demand_generator, demand_eval_generator, build_validation, BidirectionalLSTMEncoder, AttentionLayer, feedforwardNN, demand_loss

pd.set_option('display.max_columns', 30)

save_path = '/home/datamininglab/Downloads/Bicycle/JEONG/'
save_path_D = '/media/datamininglab/새 볼륨/Dataset/Bicycle/'
file_path = '/home/datamininglab/Downloads/Bicycle'

# data load
with open(os.path.join(save_path_D, 'demand_lookup.pkl'), 'rb') as f: demand_lookup = pickle.load(f)
with open(os.path.join(save_path_D, 'demand_vocab.pkl'), 'rb') as f: demand_vocab = pickle.load(f)
with open(os.path.join(save_path_D, 'demand_xy.pkl'), 'rb') as f: demand_xy = pickle.load(f)

demand_lookup = demand_lookup.drop('lookup_index', 1)
demand_lookup = demand_lookup.values

x_cols = [col for col in demand_xy.columns if '5min' in col]
demand_x = demand_xy[x_cols].values
demand_y = demand_xy[['rent_next_1hour','return_next_1hour']].values

# train, validation, test split (7:1:2)
train_length = int(len(demand_y) * 0.7)
eval_length = int(len(demand_y) * 0.1)

train_x, eval_x, test_x = demand_x[:train_length], demand_x[train_length:train_length+eval_length], demand_x[train_length+eval_length:]
train_y, eval_y, test_y = demand_y[:train_length], demand_y[train_length:train_length+eval_length], demand_y[train_length+eval_length:]
eval_x, eval_y = build_validation(eval_x, eval_y, demand_vocab)
eval_x = eval_x[:10000]
eval_y = eval_y[:10000]
test_x, test_y = build_validation(test_x, test_y, demand_vocab)
print("Train / Validation / Test length ::: {} / {} / {}".format(len(train_y), len(eval_y), len(test_y)))


# hyperparameter
VOCAB_SIZE = len(demand_vocab) # 21,228,480
EMBEDDING_SIZE = demand_lookup.shape[1] # 44
HIDDEN_SIZE = 200
BATCH_SIZE = 256
EPOCHS = 50
N_PREDICTS = train_y.shape[1]
LEARNING_RATE = 1e-5
_ALPHA = 1.0
#N_CLASSES = len(1)


tf.reset_default_graph()

model_path = "/home/datamininglab/Downloads/Bicycle/JEONG/run/"


# architecture
with tf.name_scope('placeholder'):
    batch_size = tf.placeholder(tf.int32, name='batch_size')
    input_x = tf.placeholder(tf.int32, [None, 12], name='input_x')
    input_y = tf.placeholder(tf.float32, [None, 2], name='input_y')

with tf.device("/cpu:0"):
    embed = tf.Variable(tf.constant(0.0, shape=[VOCAB_SIZE, EMBEDDING_SIZE]), trainable=False, name='embed')

with tf.name_scope('5min_to_vector'):
    embed_placeholder = tf.placeholder(tf.float32, [VOCAB_SIZE, EMBEDDING_SIZE])
    embed_init = embed.assign(embed_placeholder)
    demand_5min_embed = tf.nn.embedding_lookup(embed, input_x)

with tf.name_scope('SFNN-BiLSTM-Attention'):
    demand_SFNN = feedforwardNN(demand_5min_embed, name='SFNN')
    #demand_SFNN = fully_connected(demand_5min_embed, EMBEDDING_SIZE, activation_fn=tf.nn.tanh)
    demand_encode = BidirectionalLSTMEncoder(demand_SFNN, name='Bi-LSTM', hidden_size=HIDDEN_SIZE)
    demand_attn = AttentionLayer(demand_encode, name='SAM', hidden_size=HIDDEN_SIZE)

with tf.name_scope('prediction'):
    out = fully_connected(demand_attn, N_PREDICTS, activation_fn=tf.nn.relu)

with tf.name_scope('loss'):
    loss = demand_loss(out, input_y, _ALPHA)
    #loss = tf.reduce_sum(tf.square(out - input_y))


# test
load_model = model_path + 'demand_5min_model.ckpt-241000'

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, load_model)

sub_data = test_x[0:BATCH_SIZE]
demand_pred = sess.run(out, feed_dict={input_x:sub_data})

remaining = BATCH_SIZE * int(len(test_x)/BATCH_SIZE)
for i in range(BATCH_SIZE, len(test_x), BATCH_SIZE):
    print("{}/{}".format(i, len(test_x)))
    if i != remaining:
        sub_data = test_x[i:i+BATCH_SIZE]
        dv = sess.run(out, feed_dict={input_x:sub_data})
        demand_pred = np.concatenate((demand_pred, dv))
    else:
        sub_data = test_x[remaining:len(test_x)]
        dv = sess.run(out, feed_dict={input_x: sub_data})
        demand_pred = np.concatenate((demand_pred, dv))
