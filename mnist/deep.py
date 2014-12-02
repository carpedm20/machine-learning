from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from scipy.ndimage import convolve
import scipy.ndimage as nd
from nolearn.dbn import DBN

import numpy as np
import pandas as pd

from time import gmtime, strftime

def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
            [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 1, 0]]]
    shift = lambda x, w: convolve(x.reshape((28, 28)), mode='constant', weights=w).ravel()
    X = np.concatenate([X] + [np.apply_along_axis(shift, 1, X, vector) for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y

def rotate_dataset(X,Y):
    XX = np.zeros(X.shape)
    for index in range(X.shape[0]):
        angle = np.random.randint(-7,7)
        XX[index,:] = nd.rotate(np.reshape(X[index,:],((28,28))),angle,reshape=False).ravel()

    X = np.vstack((X,XX))
    Y = np.hstack((Y,Y))

    return X, Y

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train_d = np.asarray(train.loc[:,'pixel0':].values/ 255.0, 'float32')
test_d = np.asarray(test.values / 255.0, 'float32')

train_l = train['label'].values

train_d, train_l = nudge_dataset(train_d, train_l)
train_d, train_l = rotate_dataset(train_d, train_l)

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
