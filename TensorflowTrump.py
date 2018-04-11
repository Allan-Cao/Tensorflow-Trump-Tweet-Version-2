################
#   MAIN CODE  #
################

# Source: 
# encoding: UTF-8
# Copyright 2018 Allan Cao and Benjamin Nysetvold
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
from tensorflow.contrib import rnn
import os
import time
import math
import numpy as np
#import my_txtutils as txt
import collections
import json # Moved
import string # Moved
tf.set_random_seed(0)

'''
#filename is the name of the json file.  Must include ".json" at the end.
#directory is the directory that the txt files will be saved to.
directory="Trump/"
filename=Trump/*json #?
def twitterjson_to_txt(filename, directory):
    tweets = json.loads(open(filename).read())
    
    n = 0
    for t in tweets:
        f = open(directory+str(n)+".txt", "w")
        f.write(t['text'])
        f.close()
        n  += 1
'''

#need to import data
# os, string
textdir = "Trump/"
def get_wordlist(textdir):
    accepted_characters = """ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890!@#$%^&*()-_=+[{]};:'"\|,<.>/?`~ """
    punctuation = """!@#$%^&*()-_=+[{]};:'"\|,<.>/?`~"""
    word_list = []
    
    for filename in os.listdir(textdir):
        f = open(textdir+filename)
        tweet = f.read()
        f.close()
        filtered_tweet = ""
        for character in tweet:
            if character in accepted_characters:
                filtered_tweet = filtered_tweet + character
        for character in tweet:
            if character in punctuation:
                if character not in word_list:
                    word_list.append(character)
        translator = str.maketrans('', '', string.punctuation)
        tweet_without_punctuation = filtered_tweet.translate(translator)
        words_in_tweet = tweet_without_punctuation.split(' ')
        for word in words_in_tweet:
            if word not in word_list:
                word_list.append(word)
    return word_list

def build_dataset(word_list):
    n = 0
    dataset = {}
    for word in word_list:
        dataset[n] = word
        n += 1
    return dataset

def datasetlength(dataset):
    return len(dataset)
'''
# Define Constants
time_steps=*
#hidden LSTM units
num_units=*
#input
n_input=length of tweet
#learning rate **** CHANGE ****
learning_rate=0.001
#from mnist
n_classes=10
#size of batch
batch_size=128

def RNN(x, weights, biases):

    x = tf.reshape(x, [-1, n_input])
    x = tf.split(x,n_input,1)
    rnn_cell = rnn.BasicLSTMCell(n_hidden)
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

vocab_size = len(dictionary)

n_input = 3

n_hidden = 512 #verify
weights = {
    'out': tf.Variable(tf.random_normal(([n_hidden, vocab_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal(([vocab_size]))
}

symbols_in_key = [ [dictionary[str(training_data[i])]] for i in range(offset, offset+n_input) ]

symbols_out_onehot = np.zeros([vocab_size], dtype=float)
symbols_out_onehot[dictionary[str(training_data[offset+n_input])]] = 1.0

_, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], feed_dict=(x:symbols_in_keys, y:symbols_out_onehot))

pred = RNN(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
rnn_cell=rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])
    
'''

