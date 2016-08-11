import theano
import os
import data_loader
import similarity
import gensim
import numpy as np
from IPython import embed
from hickle import *
mqa = data_loader.DataLoader()

attention = dict()
for types in ['train', 'val']:
    story, qa = mqa.get_story_qa_data(types, 'split_plot')
    model = gensim.models.Word2Vec.load('w2v_model')

    cnt = 0
    total_story = []
    for i in xrange(len(qa)):
        qa_info = qa[i]
        error = False
        for answer in qa_info.answers:
            if len(answer) == 0: error = True
        if error == True:
            print 'error occured!'
            continue
        cnt += 1
        '''
        print 'Question >> ', qa_info.question
        print qa_info
        print qa_info.imdb_key
        '''
        stories = story[qa_info.imdb_key]
        attended_story = list()
        for i, sentence in enumerate(stories):
            attended_story.append(similarity.fixed_attention(qa_info.question, sentence, model))
        #attended_story.sort(reverse=True)
        total_story.append(np.array(attended_story))
        print np.array(total_story).shape
    if types == 'train':
        attention[types] = total_story

    if types == 'val':
        attention[types] = total_story
        '''
        for story_, score in attended_story:
            print '======================================================================'
            print 'Top. %d ' % rank
            print 'story >> ', story_
            print 'score >> ', score
            '''
embed()


