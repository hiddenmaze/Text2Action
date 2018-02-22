import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

class seq2seq_model(object):
    def __init__(self, sentence_steps, action_steps, dim_sentence, dim_char_enc, dim_gen):
        self.action_steps = action_steps
        self.dim_action = 24
        self.stddev = 0.01

        self.sentence_steps = sentence_steps
        self.dim_sentence = dim_sentence

        self.dim_char_enc = dim_char_enc
        self.dim_gen = dim_gen
        self.dim_gen_inp = dim_gen

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

        with tf.variable_scope("action2char/attention_decoder/loop_function/weights"):
            self.W_out = tf.get_variable('W_out', dtype=tf.float32,
                                         initializer=tf.random_normal([self.dim_gen, self.dim_sentence],
                                                                      stddev=self.stddev))
            self.b_out = tf.get_variable('b_out', dtype=tf.float32,
                                         initializer=tf.random_normal([self.dim_sentence], stddev=self.stddev))
            self.W_in = tf.get_variable('W_in', dtype=tf.float32,
                                        initializer=tf.random_normal([self.dim_sentence, self.dim_gen],
                                                                      stddev=self.stddev))
            self.b_in = tf.get_variable('b_in', dtype=tf.float32,
                                        initializer=tf.random_normal([self.dim_gen], stddev=self.stddev))

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
            # put the mean pose here..

        with vs.variable_scope("char2action"):
            _cell = tf.contrib.rnn.BasicLSTMCell(self.dim_gen)

            _init_state = _cell.zero_state(_num_data, dtype=tf.float32)

            _attn_state = tf.concat([_char_enc, _random], axis=2)

            _O, _S = tf.contrib.legacy_seq2seq.attention_decoder(_dec_input_list, _init_state, _attn_state, _cell,
                                                                 initial_state_attention=True,
                                                                 loop_function=self.loop_function)

        _R = []
        for _i in range(self.action_steps):
            _R.append(tf.matmul(_O[_i], _W_out) + _b_out)

        return _R, tf.transpose(tf.stack(_O), [1, 0, 2])

    def action2char(self, _action_enc, _init_input, _random, _num_data):
        with tf.variable_scope("action2char/attention_decoder/loop_function/weights", reuse=True):
            _W_out = tf.get_variable('W_out')
            _b_out = tf.get_variable('b_out')
            _W_in = tf.get_variable('W_in')
            _b_in = tf.get_variable('b_in')

        _dec_input_list = []
        for _ in range(self.sentence_steps):
            _dec_input_list.append(tf.matmul(_init_input, _W_in) + _b_in)
            # put the mean pose here..

        with vs.variable_scope("action2char"):
            _cell = tf.contrib.rnn.BasicLSTMCell(self.dim_gen)

            _init_state = _cell.zero_state(_num_data, dtype=tf.float32)

            _attn_state = tf.concat([_action_enc, _random], axis=2)

            _O, _S = tf.contrib.legacy_seq2seq.attention_decoder(_dec_input_list, _init_state, _attn_state, _cell,
                                                                 initial_state_attention=True,
                                                                 loop_function=self.loop_function)
        for _i in range(self.sentence_steps):
            _O[_i] = tf.matmul(_O[_i], _W_out) + _b_out

        return _O

    def seq2seq_loss(self, _fake_action, _real_action, _fake_char, _real_char):
        _fake_action_array = tf.transpose(tf.stack(_fake_action), [1, 0, 2])

        _fake_char_array = tf.transpose(tf.stack(_fake_char), [1, 0, 2])

        return tf.reduce_mean((_fake_action_array - _real_action) ** 2) + \
               5.0*tf.reduce_mean((_fake_char_array - _real_char) ** 2),\
               tf.reduce_mean((_fake_action_array - _real_action) ** 2),\
               tf.reduce_mean((_fake_char_array - _real_char) ** 2)

