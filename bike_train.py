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


# training
with tf.Session() as sess:
    timestamp = str(int(time.time()))
    run_dir = "/home/datamininglab/Downloads/Bicycle/JEONG/run"
    out_dir = run_dir
    print("Writing to {}\n".format(out_dir))

    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
    grads_and_vars = tuple(zip(grads, tvars))
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


    # keep track of gradient values and sparsity
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            grad_summaries.append(grad_hist_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)

    loss_summary = tf.summary.scalar('loss', loss)
    #acc_summary = tf.summary.scalar('accuracy', acc)

    train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    dev_summary_op = tf.summary.merge([loss_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    ckpt = tf.train.get_checkpoint_state(os.path.join(out_dir, './check'))

    sess.run(tf.global_variables_initializer())
    sess.run(embed_init, feed_dict={embed_placeholder : demand_lookup})

    train_loss_lst = []
    eval_loss_lst = []
    for epoch in range(EPOCHS):

        # shuffle
        indices = np.arange(len(train_x))
        np.random.shuffle(indices)
        train_x = itemgetter(*indices)(train_x)
        train_y = itemgetter(*indices)(train_y)

        print('CURRENT EPOCH {}'.format(epoch+1))

        train_batch = demand_generator(train_x, train_y, BATCH_SIZE, demand_vocab)
        num_batches = len(train_x) // BATCH_SIZE

        for itr in range(num_batches):
            x_batch, y_batch = next(train_batch)

            feed_ = {input_x: x_batch,
                     input_y: y_batch,
                     batch_size: BATCH_SIZE}
            _, step, summaries, cost = sess.run([train_op, global_step, train_summary_op, loss], feed_dict=feed_)
            train_loss_lst.append(cost)
            time_str = str(int(time.time()))
            print("Time {} / Step {} \n   Epoch [{}/{}], batch [{}/{}], loss {:g}".format(time_str, step, epoch+1, EPOCHS, itr+1, int(len(train_x)/BATCH_SIZE), cost))
            train_summary_writer.add_summary(summaries, step)

            if step % 1000 == 0:
                feed_eval = {input_x: eval_x,
                             input_y: eval_y}
                step, summaries, cost = sess.run([global_step, dev_summary_op, loss], feed_dict=feed_eval)
                eval_loss_lst.append(cost)
                print("{} evaluation {}".format("="*3,"="*3))
                print("{}, loss {:g}".format(time_str, cost))
                print("="*12)

            if step % 1000 == 0:
                saver.export_meta_graph(os.path.join(run_dir, 'demand_5min_model.meta'), collection_list=['train_var'])
                saver.save(sess, os.path.join(run_dir, 'demand_5min_model.ckpt'), global_step=global_step)
