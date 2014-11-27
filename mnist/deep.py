from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from scipy.ndimage import convolve
import scipy.ndimage as nd
from nolearn.dbn import DBN

import numpy as np
import pandas as pd
import cv2

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train_d = train.loc[:,'pixel0':].values
test_d = np.asarray(test.values / 255.0, 'float32')

train_l = train['label'].values
test_l = test['label'].values

clf = DBN([train_data.shape[1], 350, 10],
          learn_rates=0.3,
          learn_rate_decays=0.95,
          learn_rates_pretrain=0.005,
          epochs=120,
          verbose=1,
         )

clf.fit(train_d, train_l)

result = pd.read_csv("result.csv")
result.Label = clf.predict(test_l)
result.to_csv("result.csv", index_label='ImageId', col=['Label'], index=False)

