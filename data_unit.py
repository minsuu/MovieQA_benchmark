import sys
import numpy as np
sys.path.append('/data/movieQA/skip-thoughts/')
import skipthoughts
from tqdm import tqdm
from IPython import embed
#from w2v_model import *
import hickle
import threading
import pickle
#import tensorflow as tf
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from gensim_w2v import *
from multiprocessing import Process, Queue

import math
model = None
model = skipthoughts.load_model()



class Dataset(object):
    """This is a dataset ADT that contains story, QA.

    Args:
        param1 (dictionay) : (IMdb key, video clip name value) pair dictionary
        param2 (list) : QAInfo type list.

        We are able to get param1 and param2 by mqa.get_story_qa_data() function.
    """

    def __init__(self, story=None, qa=None):
        self.story = story
        self.qa = qa

        # embedding matrix Z = Word2Vec or Skipthoughts
        self.zq = [] # embedding matrix Z * questino q
        self.zsl = [] # embedding matrix Z * story sl
        self.zaj = [] # embedding matrix Z * answer aj
        self.ground_truth = [] # correct answer index

        self.zq_val = []
        self.zsl_val = []
        self.zaj_val = []
        self.ground_truth_val = []

        self.index_in_epoch_train = 0
        self.index_in_epoch_val = 0
        self.num_train_examples = 0
        self.num_val_examples = 0
        #self.embedding() # for generating hickle dump file.

    def load_dataset(self, embedding_method):
        if embedding_method == 'skip':
            skip_embed = hickle.load('./hickle_dump/skip_plot_real.hkl')
            self.zq = skip_embed['zq_train']
            self.zsl = skip_embed['zsl_train']
            self.zaj = skip_embed['zaj_train']
            self.ground_truth = skip_embed['ground_truth_train']

            print self.zq.shape
            print self.zsl.shape
            print self.zaj.shape
            print self.ground_truth.shape
            assert self.zq.shape == (9566, 1, 4800)
            assert self.zsl.shape == (9566, 6, 4800)
            assert self.zaj.shape == (9566, 5, 4800)
            assert self.ground_truth.shape == (9566, )

            self.num_train_examples = self.zq.shape[0]



    def embedding(self, embedding_method):
        """ getting Zq and Zsl by using (Word2Vec or Skipthoughts).
        """

        class DataSets(object):
            pass

        data_sets = DataSets()

        if embedding_method == 'word2vec':
            print "=============================================================================================================="
            print 'embedding process start...!'
            print "================================================================================================================="

            for qa_info in self.qa:
                """ ==================================================
                getting basic factor of dataset."""
                question = str(qa_info.question)
                answers = [str(answer) for answer in qa_info.answers]
                correct_index = qa_info.correct_index
                imdb_key = str(qa_info.imdb_key)
                stories = self.story[imdb_key]
                error = False
                imdb_key_check = dict()
                validation_flag = str(qa_info.qid)
                """================================================="""

                for answer in answers:
                    if len(answer) == 0: error = True
                if error == True : continue
                question_embedding = encode_w2v_gensim(question) # word2vec embedding : question

                local_answers = [encode_w2v_gensim(answer) for answer in answers] # word2vec embedding : answer
                local_stories = []
                if imdb_key in imdb_key_check: last_stories
                else:
                    imdb_key_check[imdb_key] = 1
                    for sentence in stories:
                        local_stories.append(encode_w2v_gensim(sentence))

                w2v_dim = 300
                if validation_flag.find('train') != -1:
                    self.zq.append(question_embedding.reshape((1,w2v_dim)))
                    self.zaj.append(np.array(local_answers))
                    self.ground_truth.append(correct_index)
                    zsl_row = np.array(local_stories).shape[0]
                    print "zsl shape >> ",
                    print np.array(local_stories).shape
                    self.zsl.append(np.array(local_stories))

                if validation_flag.find('val') != -1:
                    self.zq_val.append(question_embedding.reshape((1,w2v_dim)))
                    self.zaj_val.append(np.array(local_answers))
                    self.ground_truth_val.append(correct_index)
                    zsl_row = np.array(local_stories).shape[0]
                    print "zsl shape >> ",
                    print np.array(local_stories).shape
                    self.zsl_val.append(np.array(local_stories))


                print "===================================================="
                print "each QAInfo status >> "
                print "question embedding shape >> ",
                print np.array(self.zq).shape
                print np.array(self.zq_val).shape
                print "answer embedding shape >> ",
                print np.array(self.zaj).shape
                print np.array(self.zaj_val).shape
                print "stories embedding shape >> ",
                try:
                    print np.array(self.zsl).shape
                    print np.array(self.zsl_val).shape
                except:
                    print "warning : dimension error."
                print "ground truth embedding shape >> ", np.array(self.ground_truth).shape
                print np.array(self.ground_truth_val).shape



            w2v_dict = dict()
            w2v_dict['zq_train'] = np.reshape(np.array(self.zq), (-1,1,w2v_dim))
            w2v_dict['zaj_train'] = np.array(self.zaj)
            w2v_dict['zsl_train'] = self.zsl
            w2v_dict['ground_truth_train'] = np.array(self.ground_truth)

            w2v_dict['zq_val'] = np.reshape(np.array(self.zq_val), (-1,1,w2v_dim))
            w2v_dict['zaj_val'] = np.array(self.zaj_val)
            w2v_dict['zsl_val'] = self.zsl_val
            w2v_dict['ground_truth_val'] = np.array(self.ground_truth_val)


            self.num_train_examples = np.array(self.zq).shape[0]
            self.num_val_examples = np.array(self.zq_val).shape[0]

            return w2v_dict


        if embedding_method == 'skipthoughts':
            def embedding_thread(x, y, output):
                imdb_key_check = {}
                last_stories = []
                for i in tqdm(xrange(x,y)):
                    error = False

                    qa_info = self.qa[i]
                    question = str(qa_info.question)
                    answers = qa_info.answers
                    correct_index = qa_info.correct_index
                    imdb_key = str(qa_info.imdb_key)
                    validation_flag = str(qa_info.qid)

                    for answer in answers:
                        if len(answer) == 0 : error = True
                    if error == True :continue

                    question_embedding = skipthoughts.encode(model, [question])
                    words_in_question = word_tokenize(question)
                    assert question_embedding.shape == (1,4800)

                    local_answers = skipthoughts.encode(model, answers)


                    stories = self.story[imdb_key]

                    local_stories = []
                    if imdb_key in imdb_key_check: local_stories = last_stories
                    else:
                        imdb_key_check[imdb_key] = 1
                        local_stories = skipthoughts.encode(model, stories)
                        '''
                        for sentence in stories:
                            #local_stories.append(skipthoughts.encode(model, [sentence]))
                            paragraph_tokenize = sent_tokenize(paragraph)
                            for sentences in paragraph_tokenize:
                                words_detected = 0
                                for w in words_in_question:
                                    if sentences.find(w) != -1: words_detected += 1

                                if words_detected >= 1: local_stories.append(skipthoughts.encode(model, [sentences])) # skip embedding : story
                        '''
                        print local_stories.shape
                        last_stories = local_stories

                    skip_dim = 4800
                    if validation_flag.find('train') != - 1:
                        self.zq.append(question_embedding)
                        self.zaj.append(np.array(local_answers).reshape(5,4800))
                        self.ground_truth.append(correct_index)
                        zsl_row = np.array(local_stories).shape[0]
                        print "zsl shape >> ",
                        print np.array(local_stories).shape
                        self.zsl.append(np.array(local_stories).reshape(zsl_row,4800))

                    elif validation_flag.find('val') != -1:
                        self.zq_val.append(question_embedding)
                        self.zaj_val.append(np.array(local_answers).reshape(5,4800))
                        self.ground_truth_val.append(correct_index)
                        zsl_row = np.array(local_stories).shape[0]
                        self.zsl_val.append(np.array(local_stories).reshape(zsl_row,4800))



                    print "==========================="
                    print "each QAInfo status >> "
                    print "question embedding shape >> ",
                    print np.array(self.zq).shape
                    print np.array(self.zq_val).shape
                    print "answer embedding shape >> ",
                    print np.array(self.zaj).shape
                    print np.array(self.zaj_val).shape
                    print "stories embedding shape >> ",
                    try:
                        print np.array(self.zsl).shape
                        print np.array(self.zsl_val).shape
                    except:
                        print "warning : dimension error."

                    print "ground truth shape >> ",
                    print np.array(self.ground_truth).shape
                    print np.array(self.ground_truth_val).shape
                    print "=========================="

                output.put(self.zq)
                output.put(self.zq_val)
                output.put(self.zaj)
                output.put(self.zaj_val)
                output.put(self.zsl)
                output.put(self.zsl_val)
                output.put(self.ground_truth)
                output.put(self.ground_truth_val)

            qa_length = len(self.qa)
            worker = 36
            step_size = qa_length / worker
            procs = []
            output_q = []
            for i in xrange(worker+1):
                start = i*step_size
                end = min([(i+1)*step_size, qa_length])
                output_q.append(Queue())
                procs.append(Process(target=embedding_thread, args=(start,end,output_q[-1])))

            skip_dict = dict()

            key_list = ['zq_train', 'zq_val', 'zaj_train', 'zaj_val', 'zsl_train', 'zsl_val', 'ground_truth_train',
                        'ground_truth_val']
            skip_dict['zq_train'] = []
            skip_dict['zq_val'] = []
            skip_dict['zaj_train'] = []
            skip_dict['zaj_val'] = []
            skip_dict['zsl_train'] = []
            skip_dict['zsl_val'] = []
            skip_dict['ground_truth_train'] = []
            skip_dict['ground_truth_val'] = []
            for p in procs: p.start()
            for i in xrange(worker+1):
                zq_tmp = np.array(output_q[i].get())
                zq_tmp_val = np.array(output_q[i].get())
                zaj_tmp = np.array(output_q[i].get())
                zaj_tmp_val = np.array(output_q[i].get())
                zsl_tmp = output_q[i].get()
                zsl_tmp_val = output_q[i].get()
                gt_tmp = np.array(output_q[i].get())
                gt_tmp_val = np.array(output_q[i].get())
                skip_dict['zq_train'].extend(zq_tmp)
                skip_dict['zq_val'].extend(zq_tmp_val)
                skip_dict['zaj_train'].extend(zaj_tmp)
                skip_dict['zaj_val'].extend(zaj_tmp_val)
                skip_dict['zsl_train'].extend(zsl_tmp)
                skip_dict['zsl_val'].extend(zsl_tmp_val)
                skip_dict['ground_truth_train'].extend(gt_tmp)
                skip_dict['ground_truth_val'].extend(gt_tmp_val)

            for p in procs : p.join()

            for keys in key_list:
                if 'zsl' in keys: continue
                skip_dict[keys] = np.array(skip_dict[keys])

            return skip_dict


    def next_batch(self, batch_size, type = 'train'):
        """ at training phase, getting training(or validation) data of predefined batch size.

        Args:
            param1 (int) : batch size
            param2 (string) : type of the data you want to get. You might choose between 'train' or 'val'

        Return:
            batch size of (zq, zaj, zsl, ground_truth) pair value would be returned.
        """

        if type == 'train':
            assert batch_size <= self.num_train_examples

            start = self.index_in_epoch_train
            self.index_in_epoch_train += batch_size

            if self.index_in_epoch_train > self.num_train_examples:
                """
                if batch index touch the # of exmaples,
                shuffle the training dataset and start next new batch
                """
                perm = np.arange(self.num_train_examples)
                np.random.shuffle(perm)
                self.zq = self.zq[perm]
                self.zsl = self.zsl[perm]
                self.ground_truth = self.ground_truth[perm]
                self.zaj = self.zaj[perm]

                # start the next batch
                start = 0
                self.index_in_epoch_train = batch_size
            end = self.index_in_epoch_train
            print "start :%d, end :%d" % (start, end)
            return self.zq[start:end], self.zaj[start:end], self.zsl[start:end], self.ground_truth[start:end]

        elif type == 'val':
            assert batch_size <= self.num_val_examples

            start = self.index_in_epoch_val
            self.index_in_epoch_val += batch_size

            if self.index_in_epoch_val > self.num_val_examples:
                perm = np.arange(self.num_val_examples)
                np.random.shuffle(perm)
                self.zq_val = self.zq_val[perm]
                self.zsl_val = self.zsl_val[perm]
                self.ground_truth_val = self.ground_truth_val[perm]
                self.zaj_val = self.zaj_val[perm]

                start = 0
                self.index_in_epoch_val = batch_size
            end = self.index_in_epoch_val
            return self.zq_val[start:end], self.zaj_val[start:end], self.zsl_val[start:end], self.ground_truth_val[start:end]



