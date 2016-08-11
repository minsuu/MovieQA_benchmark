from data_unit2 import *
import data_loader
from hickle import *
import pickle
import sys

embedding_method = sys.argv[1]
embedding = None
if embedding_method == 'skip':
    embedding = 'skipthoughts'
elif embedding_method == 'word2vec':
    embedding = 'word2vec'

mqa = data_loader.DataLoader()
story, qa = mqa.get_story_qa_data('full', 'split_plot')
a = Dataset(story, qa)
a_dict = a.embedding(embedding)
filename = '/data/movieQA/hickle_dump/' + str(embedding_method) + '_full'
f = open(filename, 'w')
pickle.dump(a_dict, f)
f.close()
