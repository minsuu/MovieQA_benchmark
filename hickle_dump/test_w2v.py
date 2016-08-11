from hickle import *
import pickle
import numpy as np
import sys

origin_filename = sys.argv[1]
w2v = load(origin_filename)

"""
w2v['zq_train'] = np.reshape(w2v['zq_train'], (-1, 1 , 300))
w2v['zq_val'] = np.reshape(a['zq_val'], (-1, 1, 300))
print w2v['zq_train'].shape
print w2v['zq_val'].shape


w2v['zaj_train'] = np.reshape(w2v['zaj_train'], (-1,5,300))
w2v['zaj_val'] =np.reshape(a['zaj_val'], (-1,5,300))
print w2v['zaj_train'].shape
print w2v['zaj_val'].shape

dump(w2v, 'w2v_plot_real.hkl', 'w')
"""


a_t = [list(w2v['ground_truth_train'][i]).index(1) for i in xrange(w2v['ground_truth_train'].shape[0])]
a_v = [list(w2v['ground_truth_val'][i]).index(1) for i in xrange(w2v['ground_truth_val'].shape[0])]

print len(a_t)
print len(a_v)

w2v['ground_truth_train'] = np.array(a_t)
w2v['ground_truth_val'] = np.array(a_v)
print w2v['ground_truth_train'].shape
print w2v['ground_truth_val'].shape


fixed_num_sent = 20
edim = 300
tmp = np.zeros((len(w2v['zsl_train']), fixed_num_sent, edim))
i = 0
for story in w2v['zsl_train']:
    story =  np.reshape(story, (-1, edim))
    story = np.reshape(story, (1, -1, edim))
    num_sent = story.shape[1]
    if num_sent > fixed_num_sent:
        tmp[i,:,:] = story[:,:fixed_num_sent,:]
    elif num_sent < fixed_num_sent:
        padding = fixed_num_sent - num_sent
        pad_arr = np.zeros((1,padding,edim))
        tmp[i,:num_sent,:] = story
        tmp[i,num_sent:,:] = pad_arr
    i += 1

print tmp.shape

w2v['zsl_train'] = tmp


tmp = np.zeros((len(w2v['zsl_val']), fixed_num_sent, edim))
i = 0
for story in w2v['zsl_val']:
    story =  np.reshape(story, (-1, edim))
    story = np.reshape(story, (1, -1, edim))
    num_sent = story.shape[1]
    if num_sent > fixed_num_sent:
        tmp[i,:,:] = story[:,:fixed_num_sent,:]
    elif num_sent < fixed_num_sent:
        padding = fixed_num_sent - num_sent
        pad_arr = np.zeros((1,padding,edim))
        tmp[i,:num_sent,:] = story
        tmp[i,num_sent:,:] = pad_arr
    i += 1

print tmp.shape

w2v['zsl_val'] = tmp

dump(w2v, 'w2v_plot_final.hkl', 'w')

