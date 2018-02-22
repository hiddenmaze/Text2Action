import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'


class GAN_model(object):
    def __init__(self, sentence_steps, action_steps, dim_sentence, dim_char_enc, dim_gen, dim_dis):

        self.action_steps = action_steps
        self.dim_action = 24
        self.stddev = 0.01

        self.sentence_steps = sentence_steps
        self.dim_sentence = dim_sentence

        self.dim_char_enc = dim_char_enc
        self.dim_gen = dim_gen
        self.dim_dis = dim_dis
        self.dim_gen_inp = dim_gen

        with tf.variable_scope("char_weights"):
            self.W_c = tf.get_variable('W_c', dtype=tf.float32,
                                       initializer=tf.random_normal([self.dim_sentence, self.dim_char_enc],
                                                                    stddev=self.stddev))

            self.b_c = tf.get_variable('b_c', dtype=tf.float32, initializer=tf.random_normal([self.dim_char_enc],
                                                                                             stddev=self.stddev))

        with tf.variable_scope("char2action/attention_decoder/loop_function/weights"):
            self.W_out = tf.get_variable('W_out', dtype=tf.float32,
                                         initializer=tf.random_normal([self.dim_gen, self.dim_action],
                                                                      stddev=self.stddev))
            self.b_out = tf.get_variable('b_out', dtype=tf.float32,
                                         initializer=tf.random_normal([self.dim_action], stddev=self.stddev))

            self.W_in = tf.get_variable('W_in', dtype=tf.float32,
                                         initializer=tf.random_normal([self.dim_action, self.dim_gen],
                                                                      stddev=self.stddev))
            self.b_in = tf.get_variable('b_in', dtype=tf.float32,
                                         initializer=tf.random_normal([self.dim_gen], stddev=self.stddev))

        with tf.variable_scope("dis_var/weights"):
            self.W_r = tf.get_variable('W_r', dtype=tf.float32,
                                       initializer=tf.random_normal([self.dim_dis, 1], stddev=self.stddev))

            self.b_r = tf.get_variable('b_r', dtype=tf.float32, initializer=tf.random_normal([1], stddev=self.stddev))

    def char_encoder(self, _x, _seq_len):
        _x_split = tf.transpose(_x, [1, 0, 2])
        _x_split = tf.reshape(_x_split, [-1, self.dim_sentence])

        _x_split = tf.split(_x_split, self.sentence_steps, axis=0)
        with vs.variable_scope("char_enc_cell"):
            _cell = tf.contrib.rnn.BasicLSTMCell(self.dim_char_enc)

            _O, _ = tf.contrib.rnn.static_rnn(_cell, _x_split, dtype=tf.float32, sequence_length=_seq_len)
        return tf.transpose(tf.stack(_O), [1, 0, 2])

    def loop_function(self, _prev, _i):
        with tf.variable_scope("weights", reuse=True):
            _W_out = tf.get_variable('W_out')
            _b_out = tf.get_variable('b_out')
            _W_in = tf.get_variable('W_in')
            _b_in = tf.get_variable('b_in')

        return tf.matmul((tf.matmul(_prev, _W_out) + _b_out), _W_in) + _b_in

    def char2action(self, _char_enc, _init_input, _random, _num_data):
        with tf.variable_scope("char2action/attention_decoder/loop_function/weights", reuse=True):
            _W_out = tf.get_variable('W_out')
            _b_out = tf.get_variable('b_out')
            _W_in = tf.get_variable('W_in')
            _b_in = tf.get_variable('b_in')

        _dec_input_list = []
        for _ in range(self.action_steps):
            _dec_input_list.append(tf.matmul(_init_input, _W_in) + _b_in)

        with vs.variable_scope("char2action"):
            _cell_3 = tf.contrib.rnn.BasicLSTMCell(self.dim_gen)

            _init_state = _cell_3.zero_state(_num_data, dtype=tf.float32)

            _attn_state = tf.concat([_char_enc, _random], axis=2)

            _O, _S = tf.contrib.legacy_seq2seq.attention_decoder(_dec_input_list, _init_state, _attn_state, _cell_3,
                                                                 initial_state_attention=True,
                                                                 loop_function=self.loop_function)
        for _i in range(self.action_steps):
            _O[_i] = tf.matmul(_O[_i], _W_out) + _b_out

        return _O

    def discriminator(self, _char_seq, _action_seq, _num_data, _reuse_flag):
        with tf.variable_scope("dis_var/weights", reuse=True):
            _W_r = tf.get_variable('W_r')
            _b_r = tf.get_variable('b_r')

        _action_split = tf.transpose(_action_seq, [1, 0, 2])
        _action_split = tf.reshape(_action_split, [-1, self.dim_action])
        _action_split = tf.split(_action_split, self.action_steps, axis=0)

        with vs.variable_scope("dis_var/cell") as scope:
            _cell_4 = tf.contrib.rnn.BasicLSTMCell(self.dim_dis)

            _init_state = _cell_4.zero_state(_num_data, dtype=tf.float32)

            _attn_state = _char_seq

            if _reuse_flag == 0:
                _O, _ = tf.contrib.legacy_seq2seq.attention_decoder(decoder_inputs=_action_split,
                                                                    initial_state=_init_state,
                                                                    attention_states=_attn_state,
                                                                    initial_state_attention=True,
                                                                    cell=_cell_4)
            elif _reuse_flag == 1:
                scope.reuse_variables()
                _O, _ = tf.contrib.legacy_seq2seq.attention_decoder(decoder_inputs=_action_split,
                                                                    initial_state=_init_state,
                                                                    attention_states=_attn_state,
                                                                    initial_state_attention=True,
                                                                    cell=_cell_4)
        _result = tf.sigmoid(tf.matmul(_O[-1], _W_r) + _b_r)

        return _result

    def dis_loss(self, _real_label, _fake_label):
        return -tf.reduce_mean(tf.log(_real_label + 1e-8) + tf.log(1.0 - _fake_label + 1e-8))

    def gen_loss(self, _fake_label):
        return -tf.reduce_mean(tf.log(_fake_label + 1e-8))
