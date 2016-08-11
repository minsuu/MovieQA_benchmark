import sys
import w2v_data_engine as w2v
import numpy as np
import word2vec
from nltk.tokenize import word_tokenize
from IPython import embed
import utils
import math
#iter_num = int(sys.argv[1])

#a = w2v.w2v()
#print '[*] Loading Dataset is complete!'
#print '[*] w2v training start...!'
#gensim_model = gensim.models.Word2Vec(a.words, workers = 40, size = 100, min_count=1, iter=iter_num)
#print '[*] w2v_training end...!'
#gensim_model = gensim.models.Word2Vec.load('w2v_model_100')
gensim_model = word2vec.load('movie_plots_1364.d-300.mc1.bin')
#gensim_model.init_sims(replace=True)

#save_file = 'w2v_model'
#gensim_model.save(save_file)
ignore_word_list = ['.', ',',':', '?', "'s"]
w2v_dim = 300
word_clip_size = 25
def encode_w2v_gensim(sentence):
    #embedding = list()
    embedding = np.zeros(300)
    sentence = utils.normalize_alphanumeric(sentence.lower())
    word_list = sentence.split()
    #word_list = word_tokenize(sentence)
    word_size = 0
    for word in word_list:
        if word in ignore_word_list : continue
        try:
            embedding = embedding + gensim_model[word]
            if nan_check(embedding):
                print 'nan word >> ', word
                embed()
            word_size += 1
            #embedding.extend(list(gensim_model[word]))
        except:
            pass
            #print "KEY ERROR : " + word
            #print "Full sentence >> ",
            #print word_list

    #if word_size > word_clip_size: embedding = embedding[:word_clip_size*w2v_dim]
    #elif word_size < word_clip_size : embedding.extend([0.0]*w2v_dim*(word_clip_size-word_size))
    #print len(embedding)
    #assert len(embedding) == w2v_dim * word_clip_size
    embedding_norm = np.sum(embedding**2)
    embedding = embedding / (embedding_norm + 1e-6)
    assert embedding.shape == (300, )
    return embedding

def nan_check(arr):
    return math.isnan(np.sum(arr))

