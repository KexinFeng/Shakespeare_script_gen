import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time, os, collections
import urllib.request

import reader, utils
import pickle

##
"""
Load and process data, utility functions
"""

file_url = 'https://raw.githubusercontent.com/jcjohnson/torch-rnn/master/data/tiny-shakespeare.txt'
file_name = 'tinyshakespeare.txt'
if not os.path.exists(file_name):
    urllib.request.urlretrieve(file_url, file_name)

with open(file_name,'r') as f:
    raw_data = f.read()
    print("Data length:", len(raw_data))

vocab = set(raw_data)
vocab = utils.clean_words(vocab)
vocab_size = len(vocab)
idx_to_vocab = dict(enumerate(vocab))
raw_vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys()))
vocab_to_idx = collections.defaultdict(lambda : vocab_size-1)
vocab_to_idx.update(raw_vocab_to_idx)

data = [vocab_to_idx[c] for c in raw_data]

# Saving the objects:
saved_obj_name = "./saves/saved_obj" + '_' + file_name.split('.')[0] + '.pkl'
with open(saved_obj_name, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([vocab_size, vocab, idx_to_vocab, raw_vocab_to_idx], f)
del raw_data, raw_vocab_to_idx


def gen_epochs(n, num_steps, batch_size):
    for i in range(n):
        yield reader.ptb_iterator(data, batch_size, num_steps)

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

def train_network(g, num_epochs, num_steps = 200, batch_size = 32, verbose = True, save=False):
    tf.set_random_seed(2345)
    print('Start training...')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps, batch_size)):
            # every epoch is an iterator on get_batch().
            # The iterator is used for num_epoch times.

            training_loss = 0
            steps = 0
            training_state = None
            for X, Y in epoch:
                # epoch is an iterator returned from get_batch()
                # Each iteration gives a batch of data.

                # Each time optimizer is run, the parameters are updated once
                # by the batch of data

                steps += 1
                feed_dict={g['x']: X, g['y']: Y}
                if training_state is not None:
                    feed_dict[g['init_state']] = training_state
                training_loss_, training_state, _ = sess.run([g['total_loss'],
                                                      g['final_state'],
                                                      g['train_step']],
                                                             feed_dict)
                training_loss += training_loss_
            if verbose:
                print("Average training loss for Epoch", idx, ":", training_loss/steps)
            training_losses.append(training_loss/steps)

        if isinstance(save, str):
            g['saver'].save(sess, save)

    return training_losses


## Customed LSTM cell with LN.
# tf.contrib.rnn.static_rnn or tf.nn.dynamic_rnn will use cell = LayerNormalizaedLSTMCell as a callable, ie method.

# Layer normalization on the second dimention of rnn_input: [batch, num_class]
def ln(tensor, scope = None, epsilon = 1e-5):
    """ Layer normalizes a 2D tensor along its second axis """
    assert(len(tensor.get_shape()) == 2)
    m, v = tf.nn.moments(tensor, [1], keep_dims=True)
    if not isinstance(scope, str):
        scope = ''
    with tf.variable_scope(scope + 'layer_norm'):
        scale = tf.get_variable('scale',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(1))
        shift = tf.get_variable('shift',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(0))
    LN_initial = (tensor - m) / tf.sqrt(v + epsilon)

    return LN_initial * scale + shift


class LayerNormalizedLSTMCell(tf.nn.rnn_cell.RNNCell):
    """
    Adapted from TF's BasicLSTMCell to use Layer Normalization.
    Note that state_is_tuple is always True.
    """

    def __init__(self, num_units, forget_bias=1.0, activation=tf.nn.tanh):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation

    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):
            c, h = state

            # change bias argument to False since LN will add bias via shift
            concat = utils._linear([inputs, h], 4 * self._num_units, False) # tf.nn.rnn_cell._linear

            i, j, f, o = tf.split(concat, 4, axis=1)

            # add layer normalization to each gate
            i = ln(i, scope = 'i/')
            j = ln(j, scope = 'j/')
            f = ln(f, scope = 'f/')
            o = ln(o, scope = 'o/')

            new_c = (c * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) *
                   self._activation(j))

            # add layer_normalization in calculation of new hidden state
            new_h = self._activation(ln(new_c, scope = 'new_h/')) * tf.nn.sigmoid(o)
            new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)

            return new_h, new_state


##

"""
Final model with dropouts and Layer Normalization
"""
def build_graph(
    cell_type = None,
    num_weights_for_custom_cell = 5,
    state_size = 100,
    num_classes = vocab_size,
    batch_size = 32,
    num_steps = 200,
    num_layers = 3,
    build_with_dropout=False,
    learning_rate = 1e-4):

    reset_graph()

    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

    dropout = tf.constant(1.0)

    embeddings = tf.get_variable('embedding_matrix', [num_classes, state_size])

    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

    # if cell_type == 'Custom':
    #     cell = CustomCell(state_size, num_weights_for_custom_cell)
    if cell_type == 'GRU':
        cell = tf.nn.rnn_cell.GRUCell(state_size)
    elif cell_type == 'LSTM':
        cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
    elif cell_type == 'LN_LSTM':
        cell = LayerNormalizedLSTMCell(state_size)
    else:
        cell = tf.nn.rnn_cell.BasicRNNCell(state_size)

    if build_with_dropout:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout)

    if cell_type == 'LSTM' or cell_type == 'LN_LSTM':
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    else:
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

    if build_with_dropout:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout)

    init_state = cell.zero_state(batch_size, tf.float32)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

    #reshape rnn_outputs and y
    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    y_reshaped = tf.reshape(y, [-1])

    logits = tf.matmul(rnn_outputs, W) + b

    predictions = tf.nn.softmax(logits)

    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    return dict(
        x = x,
        y = y,
        init_state = init_state,
        final_state = final_state,
        total_loss = total_loss,
        train_step = train_step,
        preds = predictions,
        saver = tf.train.Saver()
    )
##
"""
Compare GRU, LSTM and LN_LSTM: 20 epochs, 80 step sequences.
"""
# epoch = 20
#
# g = build_graph(cell_type='GRU', num_steps=80)
# t = time.time()
# losses = train_network(g, epoch, num_steps=80, save="saves/GRU_20_epochs")
# print("It took", time.time() - t, "seconds to train for 20 epochs.")
# print("The average loss on the final epoch was:", losses[-1])
#
# g = build_graph(cell_type='LSTM', num_steps=80)
# t = time.time()
# losses = train_network(g, epoch, num_steps=80, save="saves/LSTM_20_epochs")
# print("It took", time.time() - t, "seconds to train for 20 epochs.")
# print("The average loss on the final epoch was:", losses[-1])
#
# g = build_graph(cell_type='LN_LSTM', num_steps=80)
# t = time.time()
# losses = train_network(g, epoch, num_steps=80, save="saves/LN_LSTM_20_epochs")
# print("It took", time.time() - t, "seconds to train for 20 epochs.")
# print("The average loss on the final epoch was:", losses[-1])

##

"""
Build, train and save the graph
"""
check_point_name = "saves/GRU_20_epochs" + '_' + file_name.split('.')[0]
g = build_graph(cell_type='GRU', num_steps=80)
t = time.time()
losses = train_network(g, 20, num_steps=80, save=check_point_name)
print("It took", time.time() - t, "seconds to train for 20 epochs.")
print("The average loss on the final epoch was:", losses[-1])

##

"""
Generate sequence of chars. 
Give the network a single character prompt, grab its predicted probability distribution for the next character.
"""
def generate_characters(g, checkpoint, num_chars, prompt='A', pick_top_chars=None):
    """ Accepts a current character, initial state"""

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        g['saver'].restore(sess, checkpoint)

        state = None
        current_char = vocab_to_idx[prompt]
        chars = [current_char]

        for i in range(num_chars):
            if state is not None:
                feed_dict={g['x']: [[current_char]], g['init_state']: state}
            else:
                feed_dict={g['x']: [[current_char]]}

            preds, state = sess.run([g['preds'],g['final_state']], feed_dict)

            if pick_top_chars is not None:
                p = np.squeeze(preds)
                p[np.argsort(p)[:-pick_top_chars]] = 0
                p = p / np.sum(p)
                current_char = np.random.choice(vocab_size, 1, p=p)[0]
            else:
                current_char = np.random.choice(vocab_size, 1, p=np.squeeze(preds))[0]

            chars.append(current_char)

    chars = map(lambda x: idx_to_vocab[x], chars)
    print("".join(chars))
    return("".join(chars))


## Generating the text
g = build_graph(cell_type='GRU', num_steps=1, batch_size=1)
generate_characters(g, check_point_name, 750, prompt='F', pick_top_chars=5)

