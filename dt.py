import pandas as pd
import random
from sklearn import tree
#from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
import copy

label_column = 0

def print_list(lst, cmt):
    print cmt, '='
    for row in lst:
        print row

       
def split_into_train_test_random(winedf):
    df = copy.deepcopy(winedf)
    s=set(range(len(df)))
    train_set = df
    test_set = []
    to_be_removed = []
    
    while len(test_set) <= 25:
        y = random.choice(list(s))
        s.remove(y)
        to_be_removed.append(y)
        test_set.append(list(df.iloc[y]))
    df.drop(df.index[to_be_removed], inplace=True)
    test_df = pd.DataFrame(test_set)
    labels_train = list(df[0])
    df.drop([0], 1, inplace=True)
    labels_test = list(test_df[0])
    test_df.drop([0], 1, inplace=True)
    features_train = df.values.tolist()
    features_test = test_df.values.tolist()

    #print 'len(features_train), len(labels_train), len(features_test), len(labels_test) = ', len(features_train), len(labels_train), len(features_test), len(labels_test)
    #print_list(features_train, "features_train")
    #print_list(labels_train, "features_train")
    #print_list(features_test, "features_train")
    #print_list(labels_test, "features_train")
    return features_train, labels_train, features_test, labels_test


def split_into_train_test_kfold(df):
    trains_tests_list = []
    kf = KFold(len(df), n_folds=7, shuffle = True, random_state = 2)
    for train_index, test_index in kf:
        #print('TRAIN: ', len(train_index), 'TEST: ', len(test_index))
        trains_tests_list.append([train_index, test_index])
    return trains_tests_list
    
    
def get_train_test_data(winedf, train_test_indexes):
    train_set = []
    test_set = []
    train_indexes = train_test_indexes[0]
    test_indexes = train_test_indexes[1]
    #print train_indexes
    #print test_indexes
    for i in train_indexes:
         train_set.append(winedf[i])
    
    for i in test_indexes:
        test_set.append(winedf[i])
    train_df = pd.DataFrame(train_set)
    labels_train = list(train_df[0])
    test_df = pd.DataFrame(test_set)
    labels_test = list(test_df[0])
    train_df.drop([0], 1, inplace=True)
    labels_test = list(test_df[0])
    test_df.drop([0], 1, inplace=True)
    features_train = train_df.values.tolist()
    features_test = test_df.values.tolist()
    
    return features_train, labels_train, features_test, labels_test
    
winedf = pd.read_csv('wine_data.csv', header=None)
#print winedf

#features_train, labels_train, features_test, labels_test = split_into_train_test2(winedf)

#print_list(trains_tests_list, 'trains_tests_list')


for i in range(7):
    features_train, labels_train, features_test, labels_test = split_into_train_test_random(winedf)
    #clf = AdaBoostClassifier()
    clf = tree.DecisionTreeClassifier(max_depth = 25)
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    acc = accuracy_score(pred, labels_test)
    print 'iteration', i, 'decision tree max depth 25 accuracy =', acc

    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    acc = accuracy_score(pred, labels_test)
    print 'iteration', i, 'kneighbors k = 5 accuracy =', acc

    clf = SVC(kernel="linear", C=0.025)
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    acc = accuracy_score(pred, labels_test)
    print 'iteration', i, 'SVM linear accuracy =', acc

    clf = SVC(gamma=2, C=0.025)
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    acc = accuracy_score(pred, labels_test)
    print 'iteration', i, 'SVM rbf accuracy =', acc

    clf = AdaBoostClassifier()
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    acc = accuracy_score(pred, labels_test)
    print 'iteration', i, 'adaboost accuracy =', acc

'''trains_tests_list = split_into_train_test_kfold(winedf)
for train_test_indexes in trains_tests_list:
    features_train, labels_train, features_test, labels_test = get_train_test_data(winedf.values.tolist(), train_test_indexes)
    #exit()
    #clf = AdaBoostClassifier()
    clf = SVC(kernel="linear", C=0.025)
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    #acc = accuracy_score(pred, labels_test)
    #print acc

    #acc_min_samples_split_50 = accuracy_score(pred50, labels_test)
    acc = accuracy_score(pred, labels_test)
    print acc
'''

'''clf = tree.DecisionTreeClassifier(max_depth = 25)
clf = KNeighborsClassifier(n_neighbors=51)
#clf = SVC(kernel="linear", C=0.025)
clf = SVC(gamma=2, C=0.025)
clf = SVC()
clf = AdaBoostClassifier()
clf = SVC(kernel="linear", C=0.025)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)


acc = accuracy_score(pred, labels_test)
print acc'''

#acc_min_samples_split_50 = accuracy_score(pred50, labels_test)
#acc = accuracy_score(pred, labels_test)

#print acc
