import numpy as np
import tensorflow as tf


class seq2seq_tester(object):
    def __init__(self, model, test_script, test_script_len, init_pose,
                 model_dir, sentence_steps, action_steps, dim_sentence, dim_char_enc, dim_gen, dim_random
                 ):
        self.action_steps = action_steps
        self.dim_action = 24
        self.num_data = 1

        self.model = model
        self.test_script = test_script
        self.test_script_len = test_script_len
        self.init_pose = init_pose
        self.batch_init = np.transpose(np.tile(self.init_pose, (1, self.num_data)), [1, 0])

        self.model_dir = model_dir
        self.sentence_steps = sentence_steps
        self.dim_sentence = dim_sentence
        self.dim_char_enc = dim_char_enc
        self.dim_gen = dim_gen
        self.dim_random = dim_random

        self.ph_sen = tf.placeholder(tf.float32, [None, self.sentence_steps, self.dim_sentence])
        self.ph_random = tf.placeholder(tf.float32, [None, self.sentence_steps, self.dim_random])
        self.ph_seq_len = tf.placeholder(tf.float32, [None])
        self.ph_gen_init_inp = tf.placeholder(tf.float32, [None, self.dim_action])  # put zero input here
        self.ph_num_data = tf.placeholder(tf.int32, [])

    def test(self):
        char_enc_out = self.model.char_encoder(self.ph_sen, self.ph_seq_len)
        action_gen_out = self.model.char2action(char_enc_out, self.ph_gen_init_inp,
                                                self.ph_random, self.ph_num_data)

        init = tf.global_variables_initializer()

        saver = tf.train.Saver(var_list=tf.trainable_variables())

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            np.random.seed()
            sess.run(init)

            saver.restore(sess, self.model_dir)

            curr_init_input = self.batch_init

            curr_random = np.zeros((1, self.ph_random.shape[1], self.ph_random.shape[2]))

            test_esti, test_enc = sess.run(action_gen_out, feed_dict={self.ph_sen: np.transpose(self.test_script, [0, 2, 1]),
                                                            self.ph_seq_len: self.test_script_len,
                                                            self.ph_gen_init_inp: curr_init_input,
                                                            self.ph_num_data: self.num_data,
                                                            self.ph_random: curr_random
                                                            })

            test_esti = np.transpose(np.asarray(test_esti), [2, 0, 1])

        return test_esti