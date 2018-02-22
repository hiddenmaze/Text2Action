import numpy as np
import tensorflow as tf
import random
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'


class GAN_trainer(object):
    def __init__(self, gan_model, train_script, train_script_len, train_action, init_pose,
                 num_data, batch_size, gan_model_dir, seq2seq_model_dir, dis_model_dir,
                 sentence_steps, action_steps, dim_sentence, dim_char_enc, dim_gen, dim_random,
                 restore=0, restore_path='', restore_step=0,
                 max_epoch=500, save_stride=5, gen_learning_rate=0.000002, dis_learning_rate=0.000002 #0.00001
                 ):
        self.action_steps = action_steps
        self.dim_action = 24

        self.gan_model = gan_model

        self.train_script = train_script
        self.train_script_len = train_script_len
        self.train_action = train_action
        self.init_pose = init_pose

        self.num_data = num_data
        self.batch_size = batch_size

        self.batch_init = np.transpose(np.tile(self.init_pose, (1, self.batch_size)), [1, 0])

        self.num_batch = self.num_data / batch_size
        self.gan_model_dir = gan_model_dir
        self.seq2seq_model_dir = seq2seq_model_dir
        self.dis_model_dir = dis_model_dir

        self.sentence_steps = sentence_steps
        self.dim_sentence = dim_sentence
        self.dim_char_enc = dim_char_enc
        self.dim_gen = dim_gen
        self.dim_gen_inp = self.dim_action
        self.dim_random = dim_random

        self.restore = restore
        self.restore_path = restore_path
        self.restore_step = restore_step

        self.max_epoch = max_epoch
        self.save_stride = save_stride
        self.gen_learning_rate = gen_learning_rate
        self.dis_learning_rate = dis_learning_rate

        self.ph_sen = tf.placeholder(tf.float32, [None, self.sentence_steps, self.dim_sentence])
        self.ph_action = tf.placeholder(tf.float32, [None, self.action_steps, self.dim_action])
        self.ph_action_fake = tf.placeholder(tf.float32, [None, self.action_steps, self.dim_action])
        self.ph_enc_state = tf.placeholder(tf.float32, [None, self.sentence_steps, self.dim_char_enc])
        self.ph_random = tf.placeholder(tf.float32, [None, self.sentence_steps, self.dim_random])
        self.ph_seq_len = tf.placeholder(tf.float32, [None])
        self.ph_gen_init_inp = tf.placeholder(tf.float32, [None, self.dim_gen_inp])
        self.ph_num_data = tf.placeholder(tf.int32, [])

    def train(self):
        char_enc_out = self.gan_model.char_encoder(self.ph_sen, self.ph_seq_len)
        action_gen_out = self.gan_model.char2action(self.ph_enc_state, self.ph_gen_init_inp,
                                                         self.ph_random, self.ph_num_data)

        label_real = self.gan_model.discriminator(self.ph_enc_state, self.ph_action, self.ph_num_data, 0)
        label_fake4gen = self.gan_model.discriminator(self.ph_enc_state,
                                                      tf.transpose(tf.stack(action_gen_out), [1, 0, 2]),
                                                      self.ph_num_data, 1)
        label_fake4dis = self.gan_model.discriminator(self.ph_enc_state, self.ph_action_fake,
                                                      self.ph_num_data, 1)

        dis_loss = self.gan_model.dis_loss(label_real, label_fake4dis)
        gen_loss = self.gan_model.gen_loss(label_fake4gen)

        all_vars = tf.trainable_variables()
        gen_vars = [var for var in all_vars if var.name.startswith('char2action')]
        dis_vars = [var for var in all_vars if var.name.startswith('dis_var')]

        dis_optimizer = tf.train.AdamOptimizer(learning_rate=self.dis_learning_rate).minimize(dis_loss, var_list=dis_vars)
        gen_optimizer = tf.train.AdamOptimizer(learning_rate=self.gen_learning_rate).minimize(gen_loss, var_list=gen_vars)

        seq2seq_vars = gen_vars

        init = tf.global_variables_initializer()

        gan_saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=100)
        seq2seq_saver = tf.train.Saver(var_list=seq2seq_vars)

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(init)

            seq2seq_saver.restore(sess, self.seq2seq_model_dir)

            if self.restore == 1:
                gan_saver.restore(sess, self.restore_path)
                print('Restored %s' % self.restore_path)

            for _epoch in range(self.max_epoch - self.restore_step):
                batch_shuffle = [_i for _i in range(self.num_data)]
                random.shuffle(batch_shuffle)

                for i in range(self.num_batch):
                    batch_idx = [batch_shuffle[idx] for idx in range(i * self.batch_size, (i + 1) * self.batch_size)]
                    script_batch = self.train_script[batch_idx, :, :]
                    length_batch = self.train_script_len[batch_idx]
                    action_batch = self.train_action[batch_idx, :, :]

                    curr_init_input = self.batch_init

                    curr_random = np.random.normal(size=[self.batch_size, self.ph_random.shape[1], self.ph_random.shape[2]])

                    curr_enc_out = sess.run(char_enc_out, feed_dict={self.ph_sen: np.transpose(script_batch, [0, 2, 1]),
                                                                     self.ph_seq_len: length_batch,
                                                                     })

                    curr_fake = sess.run(action_gen_out, feed_dict={self.ph_enc_state: curr_enc_out,
                                                                    self.ph_gen_init_inp: curr_init_input,
                                                                    self.ph_random: curr_random,
                                                                    self.ph_num_data: self.batch_size})

                    sess.run(dis_optimizer, feed_dict={self.ph_sen: np.transpose(script_batch, [0, 2, 1]),
                                                       self.ph_enc_state: curr_enc_out,
                                                       self.ph_action: np.transpose(action_batch, [0, 2, 1]),
                                                       self.ph_action_fake: np.transpose(np.stack(curr_fake), [1, 0, 2]),
                                                       self.ph_num_data: self.batch_size
                                                       })

                    sess.run(gen_optimizer, feed_dict={self.ph_sen: np.transpose(script_batch, [0, 2, 1]),
                                                       self.ph_enc_state: curr_enc_out,
                                                       self.ph_gen_init_inp: curr_init_input,
                                                       self.ph_action: np.transpose(action_batch, [0, 2, 1]),
                                                       self.ph_random: curr_random,
                                                       self.ph_num_data: self.batch_size
                                                       })

                    curr_dis_loss = sess.run(dis_loss, feed_dict={self.ph_sen: np.transpose(script_batch, [0, 2, 1]),
                                                                  self.ph_enc_state: curr_enc_out,
                                                                  self.ph_action: np.transpose(action_batch,
                                                                                               [0, 2, 1]),
                                                                  self.ph_action_fake: np.transpose(
                                                                       np.stack(curr_fake),
                                                                       [1, 0, 2]),
                                                                  self.ph_num_data: self.batch_size
                                                                  })

                    curr_gen_loss = sess.run(gen_loss, feed_dict={self.ph_sen: np.transpose(script_batch, [0, 2, 1]),
                                                                  self.ph_enc_state: curr_enc_out,
                                                                  self.ph_action: np.transpose(action_batch,
                                                                                            [0, 2, 1]),
                                                                  self.ph_gen_init_inp: curr_init_input,
                                                                  self.ph_random: curr_random,
                                                                  self.ph_num_data: self.batch_size
                                                                   })

                    if i % 100 == 0:
                        print(str(_epoch+self.restore_step), ': batch_gen_loss : '+str(curr_gen_loss)+', dis_loss :' + str(curr_dis_loss))

                if (_epoch + 1 + self.restore_step ) % self.save_stride == 0:
                    model_save_path = gan_saver.save(sess, self.gan_model_dir + '/model.ckpt',
                                                     global_step=_epoch + 1 + self.restore_step)
                    print("Model saved in file : %s" % model_save_path)
