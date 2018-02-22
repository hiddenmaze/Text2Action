import numpy as np
import tensorflow as tf
import random

class seq2seq_trainer(object):
    def __init__(self, model, train_script, train_script_len, train_action, init_pose,
                 num_data, batch_size, model_dir, sentence_steps, action_steps, dim_sentence, dim_char_enc, dim_gen, dim_random,
                 restore=0, restore_path='', restore_step=0,
                 max_epoch=500, save_stride=5, learning_rate=0.00005
                 ):
        self.action_steps = action_steps
        self.dim_action = 24

        self.model = model
        self.train_script = train_script
        self.train_script_len = train_script_len
        self.train_action = train_action
        self.init_pose = init_pose
        self.num_data = num_data
        self.batch_size = batch_size

        self.batch_init = np.transpose(np.tile(self.init_pose, (1, self.batch_size)), [1, 0])

        self.num_batch = self.num_data / batch_size
        self.model_dir = model_dir
        self.sentence_steps = sentence_steps
        self.dim_sentence = dim_sentence
        self.dim_char_enc = dim_char_enc
        self.dim_gen = dim_gen
        self.dim_random = dim_random

        self.restore = restore
        self.restore_path = restore_path
        self.restore_step = restore_step
        
        self.max_epoch = max_epoch
        self.save_stride = save_stride
        self.learning_rate = learning_rate
        
        self.ph_sen = tf.placeholder(tf.float32, [None, self.sentence_steps, self.dim_sentence])
        self.ph_action = tf.placeholder(tf.float32, [None, self.action_steps, self.dim_action])

        self.ph_random_a2c = tf.placeholder(tf.float32, [None, self.action_steps, self.dim_random])
        self.ph_random_c2a = tf.placeholder(tf.float32, [None, self.sentence_steps, self.dim_random])

        self.ph_seq_len = tf.placeholder(tf.float32, [None])
        self.ph_action_init_inp = tf.placeholder(tf.float32, [None, self.dim_action])
        self.ph_char_init_inp = tf.placeholder(tf.float32, [None, self.dim_sentence])
        self.ph_num_data = tf.placeholder(tf.int32, [])

    def train(self):
        char_enc_out = self.model.char_encoder(self.ph_sen, self.ph_seq_len)
        action_gen_out, action_enc_out = self.model.char2action(char_enc_out, self.ph_action_init_inp,
                                                                self.ph_random_c2a, self.ph_num_data)
        action2char_out = self.model.action2char(action_enc_out, self.ph_char_init_inp,
                                                 self.ph_random_a2c, self.ph_num_data)

        seq2seq_loss, action_loss, enc_loss = self.model.seq2seq_loss(action_gen_out, self.ph_action, action2char_out, self.ph_sen)

        seq2seq_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(seq2seq_loss)

        init = tf.global_variables_initializer()

        saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=100)

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(init)

            if self.restore == 1:
                saver.restore(sess, self.restore_path)
                print('Restored %s' % self.restore_path)

            for _epoch in range(self.max_epoch):
                batch_shuffle = [_i for _i in range(self.num_data)]
                random.shuffle(batch_shuffle)

                for i in range(self.num_batch):
                    batch_idx = [batch_shuffle[idx] for idx in range(i * self.batch_size, (i + 1) * self.batch_size)]
                    script_batch = self.train_script[batch_idx, :, :]
                    length_batch = self.train_script_len[batch_idx]
                    action_batch = self.train_action[batch_idx, :, :]

                    curr_action_init_input = self.batch_init
                    curr_char_init_input = np.zeros((self.batch_size, self.dim_sentence))

                    curr_random_c2a = np.zeros((self.batch_size, self.ph_random_c2a.shape[1], self.ph_random_c2a.shape[2]))
                    curr_random_a2c = np.zeros((self.batch_size, self.ph_random_a2c.shape[1], self.ph_random_a2c.shape[2]))

                    feed_dict = {self.ph_sen: np.transpose(script_batch, [0, 2, 1]),
                                 self.ph_seq_len: length_batch,
                                 self.ph_action: np.transpose(action_batch, [0, 2, 1]),
                                 self.ph_action_init_inp: curr_action_init_input,
                                 self.ph_char_init_inp: curr_char_init_input,
                                 self.ph_num_data: self.batch_size,
                                 self.ph_random_c2a: curr_random_c2a,
                                 self.ph_random_a2c: curr_random_a2c
                                 }
                    sess.run(seq2seq_optimizer, feed_dict=feed_dict)

                    curr_loss = sess.run(seq2seq_loss, feed_dict=feed_dict)
                    curr_action_loss = sess.run(action_loss, feed_dict=feed_dict)
                    curr_enc_loss = sess.run(enc_loss, feed_dict=feed_dict)
                    if i % 100 == 0:
                        print('current epoch : ' + str(_epoch+self.restore_step), ', current loss : ' + str(curr_loss),
                               ', ', str(curr_action_loss), ', ', str(curr_enc_loss))

                if (_epoch + 1) % self.save_stride == 0:
                    model_save_path = saver.save(sess, self.model_dir + '/model.ckpt',
                                                 global_step=_epoch + 1 + self.restore_step)
                    print("Model saved in file : %s" % model_save_path)
