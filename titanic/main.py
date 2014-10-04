from sklearn.linear_model import SGDClassifier
import csv
import numpy as np
import pandas as pd

train = pd.read_csv('data/train.csv',header=0)

# male, female

train.Gender = train.Sex.map({'female':0, 'male':1}).astype(int)

if len(train.Embarked[train.Embarked.isnull()]) > 0:
    train.Embarked[train.Embarked.isnull()] = train.Embarked.dropna().mode().values

ports = list(enumerate(np.unique(train.Embarked)))
ports_dict = {name:i for i, name in ports}
train.Embarked = train.Embarked.map(lambda x: ports_dict[x]).astype(int)

median_age = train.Age.dropna().median()
if len(train.Age[train.Age.isnull()]) > 0:
    train.Age[train.Age.isnull()] = median_age

train = train.drop(['Name','Sex','Ticket','Cabin','PassengerId'], axis=1)

test = pd.read_csv('data/test.csv', header=0)

test.Gender = test.Sex.map({'female':0, 'male':1}).astype(int)

if len(test.Embarked[test.Embarked.isnull()]) > 0:
    test.Embarked[test.Embarked.isnull()] = test.Embarked.dropna().mode().values
test.Embarked = test.Embarked.map(lambda x: ports_dict[x]).astype(int)

if len(test.Age[test.Age.isnull()]) > 0:
    test.Age[test.Age.isnull()] = median_age

if len(test.Fare[ test.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(3):
        median_fare[f] = test[test.Pclass==f+1]['Fare'].dropna().median()
    for f in range(3):
        test.loc[(test.Fare.isnull()) & (test.Pclass==f+1),'Fare'] = median_fare[f]

ids = test['PassengerId'].values
test = test.drop(['Name','Sex','Ticket','Cabin','PassengerId'], axis=1)

test_data=test.values
train_data=train.values

clf = SGDClassifier(loss="hinge")
clf = clf.fit(train_data[0::,1::],train_data[0::,0])

output = clf.predict(test_data).astype(int)

predictions_file = open("SDG.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
