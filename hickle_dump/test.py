from hickle import *
import numpy as np
skip_embed = load('skip_plot.hkl')


print "zq shape >> ",
print np.array(skip_embed['zq_val']).shape

#print "zsl_val shape >> ",

#print np.array(skip_embed['zsl_val']).shape

skip_embed['zaj_train'] = np.transpose(skip_embed['zaj_train'], (0,2,1))
skip_embed['zaj_val'] = np.transpose(skip_embed['zaj_val'], (0,2,1))

for i in xrange(len(skip_embed['ground_truth_train'])):
    t = skip_embed['ground_truth_train'][i].index(1)
    skip_embed['ground_truth_train'][i] = t

for i in xrange(len(skip_embed['ground_truth_val'])):
    t = skip_embed['ground_truth_val'][i].index(1)
    skip_embed['ground_truth_val'][i] = t


print "zaj_val shape >> ",
print np.array(skip_embed['zaj_val']).shape


print "ground_truth_val shape >> ",
print np.array(skip_embed['ground_truth_val']).shape

print skip_embed['ground_truth_val']

count = dict()
for story in skip_embed['zsl_val']:
    if story.shape[0] in count:
        count[story.shape[0]] += 1
    else: count[story.shape[0]] = 1

for keys in count.keys():
    print '{%d -> %d}' % (keys, count[keys])


#dump(skip_embed, 'skip_plot_real.hkl', 'w')




