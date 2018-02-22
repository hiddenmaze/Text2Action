from os import listdir, remove
from os.path import isfile, join
from google_drive_downloader import GoogleDriveDownloader as gdd
import gzip
import gensim

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