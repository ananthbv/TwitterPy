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


import copy

label_column = -1

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
    #if (label_column == -1):
    label_column = len(df.columns) - 1
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
    
#winedf = pd.read_csv('wine_data.csv', header=None)

winedf = pd.read_csv('winequality-white.csv', header=None, delimiter=';')
#print winedf.iloc[0]
#print winedf['quality']

#features_train, labels_train, features_test, labels_test = split_into_train_test2(winedf)

#print_list(trains_tests_list, 'trains_tests_list')

all_iterations_scores = []
models = ['DTdepth5gini','DTdepth10gini','DTdepth15gini','DTdepth20gini','DTdepth25gini','DTdepth30gini','DTdepth40gini','DTdepth50gini',
          'DTdepth5entropy','DTdepth10entropy','DTdepth15entropy','DTdepth20entropy','DTdepth25entropy','DTdepth30entropy','DTdepth40entropy','DTdepth50entropy',
             'knn2weightsuniform','knn2weightsdistance','knn3weightsuniform','knn3weightsdistance','knn5weightsuniform',
             'knn5weightsdistance','knn7weightsuniform','knn7weightsdistance','knn10weightsuniform','knn10weightsdistance',
             'adaboosttreedepth5est10gini','adaboosttreedepth5est30gini','adaboosttreedepth5est50gini','adaboosttreedepth5est100gini','adaboosttreedepth10est10gini',
             'adaboosttreedepth10est30gini','adaboosttreedepth10est50gini','adaboosttreedepth10est100gini','adaboosttreedepth15est10gini','adaboosttreedepth15est30gini',
             'adaboosttreedepth15est50gini','adaboosttreedepth15est100gini','adaboosttreedepth20est10gini','adaboosttreedepth20est30gini','adaboosttreedepth20est50gini',
             'adaboosttreedepth20est100gini','adaboosttreedepth25est10gini','adaboosttreedepth25est30gini','adaboosttreedepth25est50gini','adaboosttreedepth25est100gini',
             'adaboosttreedepth30est10gini','adaboosttreedepth30est30gini','adaboosttreedepth30est50gini','adaboosttreedepth30est100gini','adaboosttreedepth40est10gini',
             'adaboosttreedepth40est30gini','adaboosttreedepth40est50gini','adaboosttreedepth40est100gini','adaboosttreedepth50est10gini','adaboosttreedepth50est30gini',
             'adaboosttreedepth50est50gini','adaboosttreedepth50est100gini',
             'adaboosttreedepth5est10entropy','adaboosttreedepth5est30entropy','adaboosttreedepth5est50entropy','adaboosttreedepth5est100entropy','adaboosttreedepth10est10entropy',
             'adaboosttreedepth10est30entropy','adaboosttreedepth10est50entropy','adaboosttreedepth10est100entropy','adaboosttreedepth15est10entropy','adaboosttreedepth15est30entropy',
             'adaboosttreedepth15est50entropy','adaboosttreedepth15est100entropy','adaboosttreedepth20est10entropy','adaboosttreedepth20est30entropy','adaboosttreedepth20est50entropy',
             'adaboosttreedepth20est100entropy','adaboosttreedepth25est10entropy','adaboosttreedepth25est30entropy','adaboosttreedepth25est50entropy','adaboosttreedepth25est100entropy',
             'adaboosttreedepth30est10entropy','adaboosttreedepth30est30entropy','adaboosttreedepth30est50entropy','adaboosttreedepth30est100entropy','adaboosttreedepth40est10entropy',
             'adaboosttreedepth40est30entropy','adaboosttreedepth40est50entropy','adaboosttreedepth40est100entropy','adaboosttreedepth50est10entropy','adaboosttreedepth50est30entropy',
             'adaboosttreedepth50est50entropy','adaboosttreedepth50est100entropy'
          ]


for i in range(10):
    features_train, labels_train, features_test, labels_test = split_into_train_test_random(winedf)
    iteration_scores = []
    dt_train_scores = []
    dt_test_scores = []
    plt.figure() 
    plt.title('Decision tree using GINI criteria')
    plt.xlabel("Depth")
    plt.ylabel("Score")
    for depth in [5, 10, 15, 20, 25, 30, 40, 50]:
        clf = tree.DecisionTreeClassifier(max_depth = depth)
        clf.fit(features_train, labels_train)
        dt_train_scores.append(clf.score(features_train, labels_train))
        dt_test_scores.append(clf.score(features_test, labels_test))
        pred = clf.predict(features_test)
        acc = accuracy_score(pred, labels_test)
        print 'iteration', i, 'decision gini tree max depth =', depth, 'accuracy =', acc
        iteration_scores.append(acc)
    plt.plot([5, 10, 15, 20, 25, 30, 40, 50], dt_train_scores, 'o-', color="b", label="Training scores")
    plt.plot([5, 10, 15, 20, 25, 30, 40, 50], dt_test_scores, 'o-', color="g", label="Testing scores")
    plt.legend(loc="best")
    #plt.show()

    plt.figure() 
    plt.title('Decision tree using entropy criteria')
    plt.xlabel("Depth")
    plt.ylabel("Score")
    dt_train_scores = []
    dt_test_scores = []
    for depth in [5, 10, 15, 20, 25, 30, 40, 50]:
        clf = tree.DecisionTreeClassifier(max_depth = depth, criterion='entropy')
        clf.fit(features_train, labels_train)
        #print 'train score =', clf.score(features_train, labels_train)
        #print 'test score =', clf.score(features_test, labels_test)
        dt_train_scores.append(clf.score(features_train, labels_train))
        dt_test_scores.append(clf.score(features_test, labels_test))
        pred = clf.predict(features_test)
        acc = accuracy_score(pred, labels_test)
        print 'iteration', i, 'decision entropy tree max depth =', depth, 'accuracy =', acc
        iteration_scores.append(acc)
    plt.plot([5, 10, 15, 20, 25, 30, 40, 50], dt_train_scores, 'o-', color="b", label="Training scores")
    plt.plot([5, 10, 15, 20, 25, 30, 40, 50], dt_test_scores, 'o-', color="g", label="Testing scores")
    plt.legend(loc="best")
    #plt.show()

    plt.figure() 
    plt.title('kNN with uniform')
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Score")
    dt_train_scores = []
    dt_test_scores = []
    for neighbors in [2, 3, 5, 7, 10, 20, 30, 40, 50]:
        for wts in ['uniform']: #, 'distance']: 
            clf = KNeighborsClassifier(n_neighbors=neighbors, weights=wts)
            clf.fit(features_train, labels_train)
            #print 'train score =', clf.score(features_train, labels_train)
            #print 'test score =', clf.score(features_test, labels_test)
            dt_train_scores.append(clf.score(features_train, labels_train))
            dt_test_scores.append(clf.score(features_test, labels_test))
            pred = clf.predict(features_test)
            acc = accuracy_score(pred, labels_test)
            print 'iteration', i, 'kneighbors k =', neighbors, 'weights = ', wts, 'accuracy =', acc
            iteration_scores.append(acc)
    plt.plot([2, 3, 5, 7, 10, 20, 30, 40, 50], dt_train_scores, 'o-', color="b", label="Training scores")
    plt.plot([2, 3, 5, 7, 10, 20, 30, 40, 50], dt_test_scores, 'o-', color="g", label="Testing scores")
    plt.legend(loc="best")
    #plt.show()

    plt.figure() 
    plt.title('kNN with weights')
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Score")
    dt_train_scores = []
    dt_test_scores = []
    for neighbors in [2, 3, 5, 7, 10, 20, 30, 40, 50]:
        for wts in ['distance']: 
            clf = KNeighborsClassifier(n_neighbors=neighbors, weights=wts)
            clf.fit(features_train, labels_train)
            #print 'train score =', clf.score(features_train, labels_train)
            #print 'test score =', clf.score(features_test, labels_test)
            dt_train_scores.append(clf.score(features_train, labels_train))
            dt_test_scores.append(clf.score(features_test, labels_test))
            pred = clf.predict(features_test)
            acc = accuracy_score(pred, labels_test)
            print 'iteration', i, 'kneighbors k =', neighbors, 'weights = ', wts, 'accuracy =', acc
            iteration_scores.append(acc)
    plt.plot([2, 3, 5, 7, 10, 20, 30, 40, 50], dt_train_scores, 'o-', color="b", label="Training scores")
    plt.plot([2, 3, 5, 7, 10, 20, 30, 40, 50], dt_test_scores, 'o-', color="g", label="Testing scores")
    plt.legend(loc="best")
    #plt.show()
    
    
    
    '''for SVCKernel in ['linear', 'sigmoid', 'poly']:
        for Cvalue in [0.001, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0, 1000.0]:
            clf = SVC(kernel=SVCKernel, C=Cvalue)
            clf.fit(features_train, labels_train)
            pred = clf.predict(features_test)
            acc = accuracy_score(pred, labels_test)
            print 'iteration', i, 'SVM kernel =', SVCKernel, 'C =', Cvalue, 'accuracy =', acc
            
    for gamma in [0.001, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]:
        for Cvalue in [0.001, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0, 1000.0]:
                clf = SVC(kernel=SVCKernel, C=Cvalue)
                clf.fit(features_train, labels_train)
                pred = clf.predict(features_test)
                acc = accuracy_score(pred, labels_test)
                print 'iteration', i, 'SVM kernel = rbf, gamma =', gamma, 'C =', Cvalue, 'accuracy =', acc'''


    for depth in [5, 10, 15, 20, 25, 30, 40, 50]:
        for estimators in [10, 30, 50, 100]:
            clf = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=depth), n_estimators=estimators)
            clf.fit(features_train, labels_train)
            print 'train score =', clf.score(features_train, labels_train)
            print 'test score =', clf.score(features_test, labels_test)
            pred = clf.predict(features_test)
            acc = accuracy_score(pred, labels_test)
            print 'iteration', i, 'adaboost gini tree depth =', depth, 'estimators =', estimators, 'accuracy =', acc
            iteration_scores.append(acc)

    for depth in [5, 10, 15, 20, 25, 30, 40, 50]:
        for estimators in [10, 30, 50, 100]:
            clf = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=depth, criterion='entropy'), n_estimators=estimators)
            clf.fit(features_train, labels_train)
            print 'train score =', clf.score(features_train, labels_train)
            print 'test score =', clf.score(features_test, labels_test)
            pred = clf.predict(features_test)
            acc = accuracy_score(pred, labels_test)
            print 'iteration', i, 'adaboost entropy tree depth =', depth, 'estimators =', estimators, 'accuracy =', acc
            iteration_scores.append(acc)

    for iterations in [500, 1000]:
        for nnunits in [15, 20]:
            for alpha in [0.1]:
                #for dropout in [None, 0.25, 0.5, 0.75]:
                clf = Classifier(layers=[
                                        Layer("Sigmoid", units=nnunits),
                                        Layer("Softmax")
                                       ],
                                    learning_rate=alpha,
                                    n_iter=iterations,
                                    n_stable=10)
                                    #dropout_rate=None)
                clf.fit(np.array(features_train), np.array(labels_train))
                print 'train score =', clf.score(features_train, labels_train)
                print 'test score =', clf.score(features_test, labels_test)
                pred = clf.predict(np.array(features_test))
                acc = accuracy_score(list(pred), labels_test)
                print 'iteration', i, 'Neural Net iterations = {}, units = {}, learning rate = {}, accuracy = {}'.format(iterations, nnunits, alpha, acc)
                

    '''for iterations in [100, 200, 500, 1000]:
        for nnunits in [3, 5, 10, 15, 20]:
            for alpha in [0.001, 0.01, 0.1]:
                for dropout in [None, 0.25, 0.5, 0.75]:
                    clf = Classifier(layers=[
                                            Layer("Sigmoid", units=nnunits),
                                            Layer("Softmax")
                                           ],
                                        learning_rate=alpha,
                                        n_iter=iterations,
                                        dropout_rate=dropout)
                    clf.fit(np.array(features_train), np.array(labels_train))
                    pred = clf.predict(np.array(features_test))
                    acc = accuracy_score(list(pred), labels_test)
                    print 'iteration', i, 'Neural Net iterations = {}, units = {}, learning rate = {}, dropout = {}, accuracy = {}'.format(iterations, nnunits, alpha, dropout, acc)'''
                    

    '''mlp.Classifier(
    layers=[mlp.Layer(activation, units=units, **params), mlp.Layer(output)], random_state=1,
    n_iter=iterations, n_stable=iterations, regularize=regularize,
    dropout_rate=dropout, learning_rule=rule, learning_rate=alpha)'''
    
    
    #print len(iteration_scores)           
    all_iterations_scores.append(iteration_scores)
    #break

    
#print_list(all_iterations_scores, 'all scores')
scoresdf = pd.DataFrame(all_iterations_scores)

for i in range(len(scoresdf.columns)):
    l = scoresdf[i].values.tolist()
    print 'Model {}, Avg. accuracy = {}'.format(models[i], sum(l) / float(len(l)))

print "PENDING TASKS\nimplement learning curves\nimplement cross val score\n"
print "try with normalization\ndifferent epochs with neural nets\ngraphviz for DTs"

print "implement with 2nd data set \
       implement learning curves \
       implement proper model selection \
       implement cross val score \
       try with normalization \
       different epochs with neural nets \
       graphviz for DTs \
       confusion matrix \
        \
       give a try with pybrain \
       learning curves with varying training sizes \
       bias variance curves \
       Assignment paper\n"

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
