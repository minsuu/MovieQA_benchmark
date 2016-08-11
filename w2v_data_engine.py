import data_loader
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import numpy as np
import collections
import random
from IPython import embed
import tensorflow as tf
import math

class w2v(object):
    def __init__(self):
        self.ignore_word_list = ['.', ',', ':','?']
        self.mqa = data_loader.DataLoader()
        self.qa = None
        self.story = None
        self.dictionary = dict()
        self.data = []
        self.words = []
        self.train_flag = False
        self.vocabulary_size = 0
        self.data_index = 0
        self.build_dataset()

    def build_dataset(self):
        for i in [1,2]:
            f = open('/data/movieQA/Books_corpus/books_large_p' + str(i) + '.txt')
            while True:
                line = f.readline()
                if not line : break
                word_list = line.split()
                local_sentence = list()
                for word in word_list:
                    if (word in self.ignore_word_list) == False:
                        local_sentence.append(unicode(word, 'utf-8'))
                        if (word in self.dictionary) == False:
                            self.dictionary[word] = len(self.dictionary)
                self.words.append(local_sentence)
            f.close()

        train_val = ['train','val','test']
        for types in train_val:
            self.story, self.qa = self.mqa.get_story_qa_data(types, 'plot')
            for key_val in self.story.keys():
                each_story = self.story[key_val]

                for paragraph in each_story:
                    sentence_tokenize_list = sent_tokenize(paragraph)
                    for sentences in sentence_tokenize_list:
                        local_sentence = list()
                        word_list = word_tokenize(sentences)
                        for word in word_list:
                            if (word in self.ignore_word_list) == False :
                                local_sentence.append(unicode(word,'utf-8'))
                            if (word in self.ignore_word_list) == False and (word in self.dictionary) == False:
                                self.dictionary[word] = len(self.dictionary)
                        self.words.append(local_sentence)

            for qa_info in self.qa:
                question = str(qa_info.question)
                answers = [str(answer) for answer in qa_info.answers]

                question_tokenize = word_tokenize(question)
                local_question = []
                for word in question_tokenize:
                    if (word in self.ignore_word_list) == False:
                        local_question.append(unicode(word, 'utf-8'))
                    if (word in self.ignore_word_list) == False and (word in self.dictionary) == False:
                        self.dictionary[word] = len(self.dictionary)
                    self.words.append(local_question)

                for ans in answers:
                    ans_tokenize = word_tokenize(ans)
                    local_answer = []
                    for ans in ans_tokenize:
                        if (ans in self.ignore_word_list) == False:
                            local_answer.append(unicode(ans,'utf-8'))
                        if (ans in self.ignore_word_list) == False and (ans in self.dictionary) == False:
                            self.dictionary[ans] = len(self.dictionary)
                        self.words.append(local_answer)




            self.dictionary['UKN'] = len(self.dictionary)
            self.vocabulary_size = len(self.dictionary)

            print 'vocabuluary size >> %d' % len(self.dictionary)
            """
            for word in self.words:
                index = self.dictionary[word]
                self.data.append(index)
            """



    # batch of skip-gram model
    def next_batch(self,batch_size, num_skips, skip_windows):
        assert batch_size % num_skips == 0
        assert num_skips <= 2*skip_windows

        batch = np.zeros(batch_size)
        label = np.zeros((batch_size,1))
        span = 2*skip_windows + 1
        buffer = collections.deque(maxlen=span)
        for _ in xrange(span):
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index+1)%len(self.data)

        for i in xrange(batch_size/num_skips):
            target = skip_windows
            avoid = [skip_windows]

            for j in xrange(num_skips):
                while target in avoid : target = random.randint(0, span-1)
                avoid.append(target)
                batch[i*num_skips+j] = buffer[skip_windows]
                label[i*num_skips+j] = buffer[target]

            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index+1)%len(self.data)
        return batch, label


    """
    def model(self):
        self.batch_size = 128
        self.embedding_size = 300
        self.skip_window = 4
        self.num_skips = 2

        self.valid_size = 16
        self.valid_window = 100
        self.valid_examples = np.random.choice(self.valid_window, self.valid_size, replace=False)

        self.num_sampled = 64    # Number of negative examples to sample.
        self.graph = tf.Graph()

        with self.graph.as_default():

            # Input data
            self.train_inputs = tf.placeholder(tf.int32, shape=[None])
            self.train_labels = tf.placeholder(tf.int32, shape=[None,1])
            self.valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)

            # Ops and variables pinned to the CPU
            with tf.device('/gpu:0'):
                # Look up embeddings for inputs.
                self.embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
                self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)

                # construct the variables for the NCE loss
                self.nce_weights = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.embedding_size], stddev=1.0 / math.sqrt(self.embedding_size)))
                self.nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

            # Compute the average NCE loss for the batch
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.

            self.loss = tf.reduce_mean(tf.nn.nce_loss(self.nce_weights, self.nce_biases, self.embed, self.train_labels, self.num_sampled, self.vocabulary_size))

            # Construct the SGD optimizer using a learning rate of 1.0
            self.optimizer = tf.train.GradientDescentOptimizer(1e-1).minimize(self.loss)

            self.init = tf.initialize_all_variables()

        self.session.run(sess.nit)
        self.session = tf.Session(graph=self.graph)
        self.saver = tf.train.Saver()
            #self.saver.restore(self.session, './word2vec_shortcut/word2vec.ckpt')

    def train(self):
        # begin training
        num_steps = 400000

            # We must initialize all variables before we use them
        print "Initialized!"

        if self.train_flag == True:
            average_loss = 0
            for step in xrange(num_steps):
                batch_inputs, batch_labels = next_batch(batch_size, num_skips, skip_window)
                feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

                _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += loss_val

                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    print "Average loss at step %d : %lf" % (step, average_loss)

                if step % 30000 == 0:
                    print "shortcut of model saved...at %d iteration...!" % step
                    #saver.save(session, save_path + str(step) + '.ckpt')

    """
    def encode(self, word):

        """ given trained word2vec model and target words(sentecne),
        get the embedding of that word.

        Args:
            param1 (list) : target word list formatting like ['I', 'like', 'apple']

        Return:
            mean-pooled embeddings of word list
        """

        print 'encoding start!'
        self.preprocessed_word = []
        print len(word)
        for w in word:
            if w in dictionary: preprocessed_word.append(w)
            else: preprocessed_word.append('UKN')
        print 'word preprocessing...'
        word_index = np.array([self.dictionary[w] for w in preprocessed_word])

        print 'now, extract embedding...'
        embeddings = self.session.run(self.embed, feed_dict={train_inputs:word_index})

        print 'done.'
        return np.mean(embeddings, axis = 0)






