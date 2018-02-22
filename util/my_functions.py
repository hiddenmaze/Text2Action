from os import listdir, remove
from os.path import isfile, join
from google_drive_downloader import GoogleDriveDownloader as gdd
import gzip
import gensim
import numpy as np

def load_w2v(w2v_path):
    if isfile(w2v_path) == False:
        print "Start downloading Google Word2Vec data"
        gdd.download_file_from_google_drive(file_id='0B7XkCwpI5KDYNlNUTTlSS21pQmM',
                                            dest_path=w2v_path+'.gz',
                                            unzip=False)
        inF = gzip.open(w2v_path+'.gz', 'rb')
        outF = open(w2v_path, 'wb')
        outF.write( inF.read() )
        inF.close()
        outF.close()

        remove(w2v_path+'.gz')

    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    return w2v_model

def load_metadata(metadata_path):
    if isfile(metadata_path) == False:
        print "Start downloading preprocessed metadata"
        gdd.download_file_from_google_drive(file_id='1k3FJOYslo7PU3U4TyM3VFuiIgpcxMEjZ',
                                            dest_path=metadata_path,
                                            unzip=False)
    npzfile = np.load(metadata_path)

    train_action = npzfile['arr_0']
    train_script = npzfile['arr_1']
    train_length = npzfile['arr_2']
    sentence_steps = np.int(npzfile['arr_3'])    
    
    return train_action, train_script, train_length, sentence_steps  