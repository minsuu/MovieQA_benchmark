from hickle import *
import sys
filename = sys.argv[1]

a = load(filename)
print 'zq_train >> ',
print a['zq_train'].shape



print 'zsl_train >> ',
try:
    print a['zsl_train'].shape
except:
    print 'it is not np array ',
    print len(a['zsl_train'])


print 'zaj_train >> ',
print a['zaj_train'].shape

print 'gt_train >> ',
print a['ground_truth_train'].shape

print 'zq_val >> ',
print a['zq_val'].shape

print 'zsl_val >> ',
try:
    print a['zsl_val'].shape
except:
    print 'it it not np array ',
    print len(a['zsl_val'])


print 'zaj val >> ',
print a['zaj_val'].shape

print 'gt val >> ',
print a['ground_truth_val'].shape


