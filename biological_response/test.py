from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt

def main():
    dataset = genfromtxt(open('data/train.csv','r'), delimiter=',', dtype='f8')
