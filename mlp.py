#!/usr/bin/env python3

import pickle
import argparse

import tensorflow as tf
import sklearn.preprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx


def batch_norm_wrapper(X, is_training, decay, name=None):
    """
    Reference : "Batch normalization: Accelerating deep network training by reducing internal covariate shift.", https://arxiv.org/abs/1502.03167
    """
    with tf.variable_scope(name or "batch_nomalization"):
        gamma = tf.Variable(tf.ones([X.get_shape()[-1]]), 
                            trainable=True, 
                            name="gamma")
        beta = tf.Variable(tf.zeros([X.get_shape()[-1]]), 
                           trainable=True, 
                           name="beta")
        
        global_mean = tf.Variable(tf.zeros([X.get_shape()[-1]]), 
                                  trainable=False, 
                                  name="global_mean")
        global_var = tf.Variable(tf.ones([X.get_shape()[-1]]), 
                                 trainable=False,
                                 name="global_var")

        def calc_moments_in_train():
            batch_mean, batch_var = tf.nn.moments(X,[0])
            global_mean_update = tf.assign(global_mean,
                                   global_mean * decay + batch_mean * (1 - decay))
            global_var_update = tf.assign(global_var,
                                  global_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([global_mean_update, global_var_update]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        def calc_moments_in_predict():
            return global_mean, global_var

        mean, var = tf.cond(is_training, 
                            calc_moments_in_train,
                            calc_moments_in_predict
                           )
    
    return tf.nn.batch_normalization(X, mean, var, beta, gamma, 1e-3)


def flatten(x, name=None):
    with tf.variable_scope('flatten'):
        dims = x.get_shape().as_list()
        if len(dims) == 4:
            flattened = tf.reshape(
                x,
                shape=[-1, dims[1] * dims[2] * dims[3]])
        elif len(dims) == 2 or len(dims) == 1:
            flattened = x
        else:
            raise ValueError('Expected n dimensions of 1, 2 or 4.  Found:',
                             len(dims))

        return flattened

def linear(x, n_output, is_batch_norm=False, is_training=False, name=None, activation=None):
    if len(x.get_shape()) != 2:
        x = flatten(x)

    n_input = x.get_shape().as_list()[1]
    with tf.variable_scope(name or "fc"):
        W = tf.get_variable(
            name='W',
            shape=[n_input, n_output],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())

        b = tf.get_variable(
            name='b',
            shape=[n_output],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))

        h = tf.nn.bias_add(
            name='h',
            value=tf.matmul(x, W),
            bias=b)
        
        if is_batch_norm:
            h = batch_norm_wrapper(h, is_training, decay = 0.9)
        
        if activation:
            h = activation(h)

        return h, W



class MLP(object):
    def __init__(self, sess, input_dim, output_dim):

        self.sess = sess
        self.mean = None
        self.std = None
        self.min_loss = None
        self.best_accuracy = None
        
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.build_model()
        sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def load(self, weights_path, meta_path):
        self.saver.restore(self.sess, weights_path)

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        self.mean = meta['mean']
        self.std = meta['std']
        self.min_loss = meta['min_loss']
        self.input_dim = meta['input_dim']
        self.output_dim = meta['output_dim']

    def save(self, sess, weights_path, meta_path, flag_export_graph=False, graph_path=None):
        meta = {
            "mean": self.mean,
            "std": self.std,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "min_loss": self.min_loss,
            "best_accuracy": self.best_accuracy
        }

        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)

        self.saver.save(sess, weights_path, latest_filename="recent.ckpt", write_meta_graph=flag_export_graph)
        
    def build_model(self):
        with tf.variable_scope('variable'):
            X = tf.placeholder(dtype=tf.float32, 
                               shape=[None, self.input_dim],
                               name="X")
            Y = tf.placeholder(dtype=tf.float32,
                               shape=[None,2],
                               name="Y")
            
            learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
            is_training = tf.placeholder(dtype=tf.bool, name='is_training')
        
        
        
        with tf.variable_scope('mlp_model'):
            # declaration of model

            h, _ = linear(X, 64, is_batch_norm=True, is_training=is_training, name="layer_1", activation=tf.nn.relu)
            h, _ = linear(h, 64, is_batch_norm=True, is_training=is_training, name="layer_2", activation=tf.nn.relu)
            Y_pred, _ = linear(h, self.output_dim, name="layer_3")

            # optimization
            cost = tf.reduce_mean(tf.squared_difference(Y_pred, Y))
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
            init = tf.global_variables_initializer()
        
            correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
        
        self.X = X
        self.Y = Y
        self.Y_pred = Y_pred
        self.accuracy = accuracy
        self.learning_rate = learning_rate
        self.is_training = is_training
        self.cost = cost
        self.optimizer = optimizer
        self.init = init

    def predict(self, X_target):
        Y_pred = self.sess.run(self.Y_pred, feed_dict={self.X: (X_target - self.mean) / self.std,
                                                        self.is_training:False})
        return Y_pred


    def evaluate(self, X_target, Y_target):
        loss, accuracy, Y_pred = self.sess.run([self.cost, self.accuracy, self.Y_pred ], feed_dict={self.X: (X_target - self.mean) / self.std,
                                                       self.Y: Y_target,
                                                        self.is_training:False})

        return Y_pred, accuracy, loss  
    
    def train(self, X_train, Y_train, batch_size, n_epoch, learning_rate, save_dir_path, X_valid=None,
              Y_valid=None, verbose_interval=100):

        if self.min_loss is None:
            self.min_loss = 999999999
        
        X_train_tensor = tf.constant(X_train, dtype=tf.float32)
        mean, std = tf.nn.moments(X_train_tensor, axes=0)
        std = tf.sqrt(std)       
        self.mean, self.std = self.sess.run([mean, std])

        for epoch_i in range(n_epoch):
            
            rand_idx_list = np.random.permutation(range(len(X_train)))
            n_batch = len(rand_idx_list) // batch_size
            for batch_i in range(n_batch):
                rand_idx = rand_idx_list[batch_i * batch_size: (batch_i + 1) * batch_size]
                batch_x = X_train[rand_idx]
                batch_y = Y_train[rand_idx]
                self.sess.run(self.optimizer,
                              feed_dict={self.X: (batch_x - self.mean) / self.std,
                                         self.Y: batch_y,
                                         self.learning_rate: learning_rate,
                                         self.is_training:True})

            loss, accuracy, Y, Y_pred = self.sess.run([self.cost, self.accuracy, self.Y, self.Y_pred ], feed_dict={self.X: (X_valid - self.mean) / self.std,
                                                       self.Y: Y_valid,
                                                        self.is_training:False})
            if  epoch_i % verbose_interval == 0:
                print("-"*30)
                print("epoh_i : {}".format(epoch_i))
                print("curr valid loss: {}, curr valid accuracy: {}, best valid loss: {}, best valid accuracy : {}".format(loss, accuracy, self.min_loss, self.best_accuracy))

            if loss < self.min_loss:
                
                self.min_loss = loss
                self.best_accuracy = accuracy
                
                weights_path = "{}/weights".format(save_dir_path)
                meta_path = "{}/meta_data.pickle".format(save_dir_path)
                self.save(self.sess, weights_path=weights_path, meta_path=meta_path)
                print("*"*30)
                print("curr loss: {}, curr accuracy: {}, best_loss: {}, best_accuracy : {}".format(loss, accuracy, self.min_loss, self.best_accuracy))     
                print("save current model")

        return self.sess


def run_train(train_path, config_path):
    # train_path = "data/preprocessed_train.csv"
    train_df = pd.read_csv(train_path)
    X_train_df = train_df.drop(["target"], axis=1)
    Y_train_df = train_df["target"]
    
    X_all= X_train_df.as_matrix()
    Y_all = Y_train_df.as_matrix()
    
    # Y to one hot 
    n_class = 2
    n_sample = len(Y_all)

    tmp = np.zeros((n_sample, n_class))
    tmp[np.arange(n_sample), Y_all] = 1
    Y_all = tmp

    # data
    rand_idx = np.random.permutation(range(len(X_all)))
    X_all = X_all[rand_idx]
    Y_all = Y_all[rand_idx]

    train_ratio = 0.8
    valid_ratio = 0.1
    test_ratio = 0.1

    data_num = len(X_all)
    train_data_num = round(data_num * train_ratio)
    valid_data_num = round(data_num * valid_ratio)
    test_data_num = round(data_num * test_ratio)

    X_train = X_all[:train_data_num]
    Y_train = Y_all[:train_data_num]
    X_valid = X_all[train_data_num:train_data_num + valid_data_num]
    Y_valid = Y_all[train_data_num:train_data_num + valid_data_num]
    X_test = X_all[train_data_num + valid_data_num:]
    Y_test = Y_all[train_data_num + valid_data_num:]

    input_dim = len(X_train[0])
    output_dim = 2
    print("input dimension : {}, output dimension : {} ".format( input_dim, output_dim))
    print("X_train shape : {}\nY_train shape : {}".format(np.shape(X_train), np.shape(Y_train)))
    print("X_valid shape : {}\nY_valid shape : {}".format(np.shape(X_valid), np.shape(Y_valid)))
    
    
    sess = tf.Session()
    mlp = MLP(sess, input_dim, output_dim)
    mlp.train(X_train, Y_train, X_valid=X_valid, Y_valid=Y_valid,
                 batch_size=64, n_epoch=3000, learning_rate = 0.005, save_dir_path='./model', verbose_interval=300)


    result_dict = {
        'model' : mlp,
        'mean' : mlp.mean,
        'std' : mlp.std, 
        'X_all' :X_all,
        'Y_all' :Y_all,
        'X_train' : X_train,
        'Y_train' : Y_train,
        'X_valid' : X_valid,
        'Y_valid' : Y_valid,
        'X_test' : X_test,
        'Y_test' : Y_test,
    }
    
    return result_dict


def run_predict(target_path, out_path, model_path, meta_path):
    X_target_df = pd.read_csv(target_path)
    try :
        X_target_df = X_target_df.drop(["target"], axis=1)
    except :
        pass

    X_target = X_target_df.as_matrix()
    
    sess = tf.Session()
    mlp = MLP(sess, 5, 2)
    mlp.load(model_path, meta_path)
    result_onehot = mlp.predict(X_target)
    result_label = np.argmax(result_onehot, axis=1)

    result = { 'target' : result_label}
    result_df = pd.DataFrame(result)
    result_df.to_csv(out_path, index=False)
    

def run_evaluate(test_path, out_path, model_path, meta_path):
    X_test_df = pd.read_csv(test_path)
    Y_test_df = X_test_df["target"]
    X_test_df = X_test_df.drop(["target"], axis=1)

    X_test = X_test_df.as_matrix()
    Y_test = Y_test_df.as_matrix()
    # Y to one hot 
    n_class = 2
    n_sample = len(Y_test)

    tmp = np.zeros((n_sample, n_class))
    tmp[np.arange(n_sample), Y_test] = 1
    Y_test = tmp
    
    sess = tf.Session()
    mlp = MLP(sess, 5, 2)
    mlp.load(model_path, meta_path)
    Y_pred, accuracy, loss = mlp.evaluate(X_test,Y_test)
    result_label = np.argmax(Y_pred, axis=1)

    result = { 'target' : result_label}
    result_df = pd.DataFrame(result)
    result_df.to_csv(out_path, index=False)
    with open("{}_metric.txt".format(out_path),"wt") as f:
        f.write("loss : {}, accuracy : {} ".format(loss,accuracy))
    
    return Y_pred, accuracy, loss  


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('mode', metavar='mode', type=str, help="mode 'train' or 'predict' or 'evaluate'" )

    # train args
    parser.add_argument('--train_path', metavar='train_path', type=str, help='path of train csv file')
    parser.add_argument('--config_path', metavar='config_path', type=str, help='path of config file')

    # predict args

    parser.add_argument('--target_path', metavar='target_path', type=str, help='path of target csv file')
    parser.add_argument('--out_path', metavar='out_path', type=str, help='path of predict result file')
    parser.add_argument('--model_path', metavar='model_path', type=str, help='path of model')
    parser.add_argument('--meta_path', metavar='meta_path', type=str, help='path of meta path')

    a = parser.parse_args()

    if a.mode == 'train':

        print("*"*30)
        print("train start")
        print("*"*30)
        run_train(a.train_path, a.config_path)
    elif a.mode == 'predict':
        print("*"*30)
        print("predict start")
        print("*"*30)
        run_predict(a.target_path, a.out_path, a.model_path, a.meta_path)
        print("predict done")
        print("-"*30)
    elif a.mode == 'evaluate': 
        print("*"*30)
        print("evalueate start")
        print("*"*30)
        Y_pred, accuracy, loss  = run_evaluate(a.target_path, a.out_path, a.model_path, a.meta_path)
        print("accuracy : {} , loss : {}".format(accuracy, loss))
        print("evalueate done")
        print("-"*30)





if __name__=='__main__':

    main()


