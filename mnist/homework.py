from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from scipy.ndimage import convolve

import scipy.ndimage as nd
from nolearn.dbn import DBN
import numpy as np
import pandas as pd

from time import gmtime, strftime
import cPickle

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train_d = np.asarray(train.loc[:,'pixel0':].values/ 255.0, 'float32')
train_l = train['label'].values

zipped_fours = [i for i in zip(train_d, train_l) if i[1] == 4]
fours = [i[0] for i in zipped_fours]

#with open(r"clf.pkl","wb") as f:
#   cPickle.dump(clf, f)

with open(r"clf.pkl","rb") as f:
   clf = cPickle.load(f)

results = []

for four in fours:
    results.append(np.dot(four,clf.net_.weights[0].as_numpy_array()))

sum_array = [0]*350
for result in results:
    for (idx, x) in enumerate(result):
        sum_array[idx] += x

avg_array = [0]*350
for (idx, x) in enumerate(sum_array):
    avg_array[idx] = x / 4072.0

avg_w_idx = [[idx,x] for (idx, x) in enumerate(avg_array)]

from operator import itemgetter
sorted_avg = sorted(avg_w_idx, key=itemgetter(1))
sorted_avg.reverse()

for x in sorted_avg[:5]:
    print x[0], "\t", x[1]
