import numpy as np
import os
import scipy.io as scio
from my_functions import load_w2v

def preprocess():
    w2v_path = '../data/GoogleNews-vectors-negative300.bin'
    w2v_model = load_w2v(w2v_path)

    embed_size = w2v_model['woman'].shape[0]

    # Load data
    f = open('../data/total_script.txt', 'r')  # put your data here
    lines = f.readlines()

    max_length = 0
    for line in lines:
        line = line.lower()
        words = line.split()

        word_cnt = 0
        for word in words:
            word_cnt += 1
        if max_length < word_cnt:
            max_length = word_cnt
    f.close()
    print('The maximum sentence length is %d' % max_length)

    pose_path = '../data/pose/'
    script_path = '../data/script/'
    pose_files = [f for f in os.listdir(pose_path) if os.path.isfile(os.path.join(pose_path, f))]
    script_files = [f for f in os.listdir(script_path) if os.path.isfile(os.path.join(script_path, f))]

    total_pose_list = []
    total_script_list = []
    vocab_keys = w2v_model.vocab.keys()
    for idx, p_file in enumerate(pose_files):
        print 'processing %dth data...' %idx
        curr_pred = scio.loadmat(pose_path+p_file)['pred_vector']
        curr_script_path = script_path + 'script_' + p_file[5:9] + '.txt'
        s_f = open(curr_script_path, 'r')  

        lines = s_f.readlines()
        for line in lines:
            line = line.lower()
            words = line.split()
            tmp_word_array = np.zeros((embed_size, len(words)))
            for word in words:
                if word not in vocab_keys:
                    curr_emb_vec = np.zeros((embed_size, ))
                else:
                    curr_emb_vec = w2v_model[word]
                tmp_word_array[:, words.index(word)] = curr_emb_vec

            total_script_list.append(tmp_word_array)
            total_pose_list.append(curr_pred)
        s_f.close()

    pose_array = np.asarray(total_pose_list)
    print pose_array.shape

    num_data = pose_array.shape[0]

    script_array = np.zeros((num_data, embed_size, max_length))
    script_length = np.zeros((num_data))

    for i in range(num_data):
        tmp_script = total_script_list[i]
        script_length[i] = total_script_list[i].shape[1]
        script_array[i, :, :total_script_list[i].shape[1]] = total_script_list[i]

    print('DATA READY')

    np.savez('../data/metadata.npz', pose_array, script_array, script_length, max_length)
    print('DATA SAVED')


