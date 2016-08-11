from hickle import *
import numpy as np
a = load('w2v_plot.hkl')

train_val  = ['ground_truth_train', 'ground_truth_val']

for val in train_val:
    for i in xrange(len(a[val])):
        a[val][i] = a[val][i][0]
        break
    print a[val]
    a[val] = np.array(a[val])
    print a[val].shape


#dump(a, 'w2v_plot_real.hkl', 'w')

