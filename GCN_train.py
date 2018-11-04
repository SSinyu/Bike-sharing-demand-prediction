import os
import time
import pickle
import argparse
import numpy as np
import tensorflow as tf
from operator import itemgetter
from bicycle_util import GCN_dataloader, GCN_generator
from model import GCN_demand

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", dest="data_path", default="/Dataset/Bicycle/", help="data save,load path")
parser.add_argument("--run_path", dest="run_path", default="/Dataset/Bicycle/run/", help="run path")
parser.add_argument("--run_dir", dest="run_dir", default='test/', help="run dir")
parser.add_argument("--data_name", dest="data_name", default="split_1hour_24input_gcn_2.pkl", help="dataset name")
parser.add_argument("--input_length", dest="input_length", type=int, default=24, help="train time interval length (default:24)")
parser.add_argument("--hidden_size", dest="hidden_size", type=int, default=300, help="dimensionality of hidden layer (default:300)")
parser.add_argument("--epochs", dest="epochs", type=int, default=50, help="number of training epoch (default:50)")
parser.add_argument("--num_checkpoints", dest="num_checkpoints", type=int, default=10, help="number of checkpoints of store (default:10)")
parser.add_argument("--dropout_rate", dest="dropout_rate", type=float, default=.3, help="dropout rate (default:0.3)")
parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=1e-3, help="learning rate (default:1e-3)")
parser.add_argument("--decay_rate", dest="decay_rate", type=float, default=0.9, help="learning rate decay rate (default:0.9)")
parser.add_argument("--val_iter", dest="val_iter", type=int, default=100, help="one validation per  several iteration")
parser.add_argument("--grad_clip", dest="grad_clip", type=int, default=10, help="gradient clip for prevent gradient explode")

args = parser.parse_args()

# data load
'''
demand_data = GCN_dataloader(dataset_='demand_lookup_1hour_2.pkl')
# save
save_path_D = '/Dataset/Bicycle/'
with open(os.path.join(save_path_D, 'split_1hour_24input_gcn_2.pkl'), 'wb') as f: pickle.dump(demand_data, f)
'''
save_path = os.path.join(args.run_path, args.run_dir)
if not os.path.exists(save_path):
    os.makedirs(save_path)

with open(os.path.join(args.data_path, args.data_name),'rb') as f: demand_data = pickle.load(f)

demand_train_x = demand_data['train_x']
demand_train_y = demand_data['train_y']
demand_test_x = demand_data['test_x']
demand_test_y = demand_data['test_y']

n_station = demand_train_x[0].shape[0]
print("====== Data Load Finished ======")

# training
with tf.Session() as sess:
    demand_gcn = GCN_demand(n_station, args.input_length, args.hidden_size, 1, args.dropout_rate)

    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.square(demand_gcn.graph3 - demand_gcn.input_y), name="loss_mse")

    timestamp = str(int(time.time()))
    print("Writing to {}\n".format(save_path))

    global_step = tf.Variable(0, trainable=False)
    learning_rate_ = tf.train.exponential_decay(args.learning_rate, global_step, 2000-args.input_length-1, args.decay_rate, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate_)

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), args.grad_clip)
    grads_and_vars = tuple(zip(grads, tvars))
    #train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    train_op = optimizer.minimize(loss, global_step=global_step)

    # keep track of gradient values and sparsity
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            grad_summaries.append(grad_hist_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)

    loss_summary = tf.summary.scalar('loss', loss)

    train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(save_path, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    dev_summary_op = tf.summary.merge([loss_summary])
    dev_summary_dir = os.path.join(save_path, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    checkpoint_dir = os.path.abspath(os.path.join(save_path, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    #saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
    saver = tf.train.Saver(max_to_keep=args.num_checkpoints)

    ckpt = tf.train.get_checkpoint_state(os.path.join(save_path, './check'))

    sess.run(tf.global_variables_initializer())

    def train_step(x_batch, y_batch):
        feed = {demand_gcn.input_x: x_batch,
                demand_gcn.input_y: y_batch}
        _, step, summaries, cost = sess.run([train_op, global_step, train_summary_op, loss], feed_dict=feed)
        time_str = str(int(time.time()))
        train_summary_writer.add_summary(summaries, step)
        return time_str, step, cost

    def dev_step(writer):
        dev_loss = []
        for x_batch, y_batch in zip(demand_test_x, demand_test_y):
            feed = {demand_gcn.input_x: x_batch, demand_gcn.input_y: y_batch}
            step, summaries, cost = sess.run([global_step, dev_summary_op, loss], feed_dict=feed)
            dev_loss.append(cost)
        cost_avg = np.average(dev_loss)
        time_str = str(int(time.time()))
        print("=== validation result \n{}, loss {:.6f}".format(time_str, cost_avg))
        writer.add_summary(summaries, step)

    for epoch in range(args.epochs):
        print("CURRENT EPOCH {}".format(epoch+1))
        num_itrs = len(demand_train_x)

        # shuffle
        ind = np.arange(num_itrs)
        np.random.shuffle(ind)
        demand_train_x = itemgetter(*ind)(demand_train_x)
        demand_train_y = itemgetter(*ind)(demand_train_y)

        for itr, (x_batch, y_batch) in enumerate(zip(demand_train_x, demand_train_y)):
            train_time, step, train_loss = train_step(x_batch, y_batch)
            if step % 10 == 0:
                print("Time {} / Step {} ::: \n   Epoch [{}/{}], batch [{}/{}], loss {:.6f}".format(train_time, step, epoch + 1, args.epochs, itr + 1, num_itrs, train_loss))
                if step % args.val_iter == 0:
                    dev_step(dev_summary_writer)
                if step % 1000 == 0:
                    saver.export_meta_graph(os.path.join(save_path, 'demand_1hour_model.meta'), collection_list=['train_var'])
                    saver.save(sess, os.path.join(save_path, 'demand_1hour_model.ckpt'), global_step=global_step)

        adj_matrix = sess.run([demand_gcn.adj_matrix], feed_dict={demand_gcn.input_x: x_batch, demand_gcn.input_y: y_batch})
        np.save(os.path.join(save_path, "adj_matrix_{}ep.npy".format(epoch+1)), adj_matrix)
