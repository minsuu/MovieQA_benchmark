from data_unit import *
import data_loader
from hickle import *
import pickle
import sys
from IPython import embed


embedding_method = sys.argv[1]
embedding = None
if embedding_method == 'skip':
    embedding = 'skipthoughts'
elif embedding_method == 'word2vec':
    embedding = 'word2vec'

mqa = data_loader.DataLoader()
story, qa = mqa.get_story_qa_data('train', 'split_plot')
a = Dataset(story, qa)
a_dict = a.embedding(embedding)

story, qa = mqa.get_story_qa_data('val', 'split_plot')
b = Dataset(story, qa)

a_dict = a.embedding(embedding)
b_dict = b.embedding(embedding)
a_dict['zq_val'] = b_dict['zq_val']
a_dict['zsl_val'] = b_dict['zsl_val']
a_dict['zaj_val'] = b_dict['zaj_val']
a_dict['ground_truth_val'] = b_dict['ground_truth_val']

filename = '/data/movieQA/' + str(embedding_method)
embed()
'''
f = open(filename, 'w')
pickle.dump(a_dict, f)
f.close()
'''
