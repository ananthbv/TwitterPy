import pandas as pd
import numpy as np
import random
from sklearn import tree
#from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sknn.mlp import Classifier, Layer
from matplotlib import pyplot as plt
import sys
import copy

def print_list(lst, cmt):
    print cmt, '='
    for row in lst:
        print row

def split_into_train_test(fulldf):
    df = copy.deepcopy(fulldf)
    s=set(range(len(df)))
    train_set = df
    test_set = []
    to_be_removed = []
    #if (label_column == -1):
    test_set_len = int(0.2 * len(df))

    while len(test_set) <= test_set_len:
        y = random.choice(list(s))
        s.remove(y)
        to_be_removed.append(y)
        test_set.append(list(df.iloc[y]))
    df.drop(df.index[to_be_removed], inplace=True)

    test_df = pd.DataFrame(test_set)
    return df, test_df

def get_features_labels_from_test_df(test_df, label_column):
    #label_column = len(test_df.columns) - 1
    labels_test = list(test_df[label_column])
    test_df.drop([label_column], 1, inplace=True)
    features_test = test_df.values.tolist()
    return features_test, labels_test

def split_train_into_folds(winedf):
    df = copy.deepcopy(winedf)
    s=set(range(len(df)))
    train_set = df
    test_set = []
    to_be_removed = []
    print 'label_column =', label_column
    test_set_len = int(0.3 * len(df))
    print 'full set len, test_set_len =', len(df), test_set_len

    while len(test_set) <= test_set_len:
        y = random.choice(list(s))
        s.remove(y)
        to_be_removed.append(y)
        test_set.append(list(df.iloc[y]))
    df.drop(df.index[to_be_removed], inplace=True)

    test_df = pd.DataFrame(test_set)
    labels_train = list(df[label_column])
    df.drop([label_column], 1, inplace=True)
    labels_test = list(test_df[label_column])
    test_df.drop([label_column], 1, inplace=True)
    features_train = df.values.tolist()
    features_test = test_df.values.tolist()

    return features_train, labels_train, features_test, labels_test
    
filename = sys.argv[1]
alg = sys.argv[2]
label_column = int(sys.argv[3])
delim = ','
if len(sys.argv) > 4:
    delim = sys.argv[4]

winedf = pd.read_csv(filename, header=None, delimiter=delim)
if (label_column == -1):
    label_column = len(winedf.columns) - 1

all_iterations_scores = []
models = []
classifiers = {}
accuracies = {}

knn = False
dt = False
svm = False
nn = False
boost = False

if alg == 'knn':
    knn = True
    for neighbors in [2, 3, 5, 7, 10, 20, 30, 40, 50]:
        for wts in ['uniform', 'distance']:
            desc = 'knn ' + str(neighbors) + ' neighbors, weights ' + str(wts)
            classifiers[desc] = KNeighborsClassifier(n_neighbors=neighbors, weights=wts)

if alg == 'dt':
    dt = True
    for depth in [5, 10, 15, 20, 25, 30, 40, 50]:
        desc = 'dt ' + 'max depth ' + str(depth) + ', criterion gini'
        classifiers[desc] = tree.DecisionTreeClassifier(max_depth = depth)
    for depth in [5, 10, 15, 20, 25, 30, 40, 50]:
        desc = 'dt ' + 'max depth ' + str(depth) + ', criterion entropy'
        classifiers[desc] = tree.DecisionTreeClassifier(max_depth = depth, criterion='entropy')

if alg == 'nn':
    nn = True
    for iterations in [500, 1000]:
        for nnunits in [15, 20]:
            for alpha in [0.1]:
                #for dropout in [None, 0.25, 0.5, 0.75]:
                desc = 'nn ' + str(iterations) + 'epochs, ' + str(nnunits) + ' units, learning rate ' + str(alpha)
                classifiers[desc] = Classifier(layers=[
                                        Layer("Sigmoid", units=nnunits),
                                        Layer("Softmax")
                                       ],
                                    learning_rate=alpha,
                                    n_iter=iterations,
                                    n_stable=10)
                
if alg == 'svm':
    svm = True
    for c in [0.001, 0.01, 0.1, 0.5, 1.0, 10.0, 100.0, 1000.0]:
        desc = 'svm linear kernel, ' + 'c '+ str(c)
        classifiers[desc] = SVC(kernel='linear', C=c)
    for SVCkernel in ['poly', 'rbf', 'sigmoid']:
        for g in [0.001, 0.01, 0.1, 0.5, 1.0, 10.0, 100.0, 1000.0]:
            for c in [0.001, 0.01, 0.1, 0.5, 1.0, 10.0, 100.0, 1000.0]:
                desc = 'svm ' + SVCkernel + ' kernel, gamma ' + str(g) + ', c '+ str(c)
                classifiers[desc] = SVC(kernel=SVCkernel, C=c, gamma = g)
if alg == 'boost':
    boost = True
    for depth in [5, 10, 15, 20, 25, 30, 40, 50]:
        for estimators in [10, 30, 50, 100]:
            for c in ['gini', 'entropy']:
                desc = 'adaboost dt max depth ' +  str(depth) + ', estimators ' + str(estimators) + ', criterion ' + c
                classifiers[desc] = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=depth, criterion=c), n_estimators=estimators)

models = classifiers.keys()

print len(winedf)
train_df, test_df = split_into_train_test(winedf)
X_test, y_test = get_features_labels_from_test_df(test_df, label_column)
print len(train_df), len(test_df)

for i in range(2):
    features_train, labels_train, features_test, labels_test = split_train_into_folds(train_df)
    iteration_scores = []
    
    for desc, clf in classifiers.iteritems():
        if (nn):
            clf.fit(np.array(features_train), np.array(labels_train))
            pred = clf.predict(np.array(features_test))
            acc = accuracy_score(list(pred), labels_test)
        else:
            clf.fit(features_train, labels_train)
            pred = clf.predict(features_test)
            acc = accuracy_score(pred, labels_test)
        iteration_scores.append(acc)
        print 'iteration', i, 'classfier description =', desc, 'accuracy =', acc            

    all_iterations_scores.append(iteration_scores)


#print_list(all_iterations_scores, 'all scores')
scoresdf = pd.DataFrame(all_iterations_scores)
best_acc = 0.0
best_clf = None
best_clf_desc = ''

for i in range(len(scoresdf.columns)):
    l = scoresdf[i].values.tolist()
    avgacc = sum(l) / float(len(l))
    print 'Model {}, Avg. accuracy = {}'.format(models[i], sum(l) / float(len(l)))
    if avgacc > best_acc:
        best_acc = avgacc
        best_clf = classifiers[models[i]]
        best_clf_desc = models[i]

print 'best accuracy =', best_acc
pred = best_clf.predict(X_test)
acc = accuracy_score(pred, y_test)
print 'Best classifier =', best_clf_desc
print 'Accuracy of best classifier on test set =', acc
#if (dt):
#    tree.export_graphviz(best_clf, out_file='tree.dot')

#print "PENDING TASKS\nimplement learning curves\nimplement cross val score\n"
#print "try with normalization\ndifferent epochs with neural nets\ngraphviz for DTs"

print "done - implement with 2nd data set \n \
implement learning curves \n \
learning curves with varying training sizes \n \
done - implement proper model selection \n \
done - implement cross val score \n \
try with normalization \n \
done - different epochs with neural nets \n \
done - graphviz for DTs - probably no use as lots of attributes\n \
confusion matrix \n \
give a try with pybrain \n \
bias variance curves \n \
Assignment paper\n"
