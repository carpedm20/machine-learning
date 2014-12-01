from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from scipy.ndimage import convolve
import scipy.ndimage as nd
from nolearn.dbn import DBN

import numpy as np
import pandas as pd

from time import gmtime, strftime

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train_d = np.asarray(train.loc[:,'pixel0':].values/ 255.0, 'float32')
test_d = np.asarray(test.values / 255.0, 'float32')

train_l = train['label'].values

clf = DBN([train_d.shape[1], 350, 10],
          learn_rates=0.3,
          learn_rate_decays=0.95,
          learn_rates_pretrain=0.005,
          epochs=120,
          verbose=1,
         )

clf.fit(train_d, train_l)
pred = clf.predict(test_d)
new_pred = pred.round()

idx = range(1,len(pred) + 1)
df = pd.DataFrame({'ImageId': idx,
                   'Label': new_pred.tolist()})
df.to_csv("result_%s.csv" % strftime("%Y-%m-%d-%H:%M:%S", gmtime()), index=False)

clf.predict(test_d[12:13])
