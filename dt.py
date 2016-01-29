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

winedf = pd.read_csv('winequality-white.csv', header=None) #, delimiter=';')
#print winedf.iloc[0]
#print winedf['quality']

#features_train, labels_train, features_test, labels_test = split_into_train_test2(winedf)

#print_list(trains_tests_list, 'trains_tests_list')

all_iterations_scores = []
models = ['DTdepth5','DTdepth10','DTdepth15','DTdepth20','DTdepth25','DTdepth30','DTdepth40','DTdepth50',
             'knn2weightsuniform','knn2weightsdistance','knn3weightsuniform','knn3weightsdistance','knn5weightsuniform',
             'knn5weightsdistance','knn7weightsuniform','knn7weightsdistance','knn10weightsuniform','knn10weightsdistance',
             'adaboosttreedepth5est10','adaboosttreedepth5est30','adaboosttreedepth5est50','adaboosttreedepth5est100','adaboosttreedepth10est10',
             'adaboosttreedepth10est30','adaboosttreedepth10est50','adaboosttreedepth10est100','adaboosttreedepth15est10','adaboosttreedepth15est30',
             'adaboosttreedepth15est50','adaboosttreedepth15est100','adaboosttreedepth20est10','adaboosttreedepth20est30','adaboosttreedepth20est50',
             'adaboosttreedepth20est100','adaboosttreedepth25est10','adaboosttreedepth25est30','adaboosttreedepth25est50','adaboosttreedepth25est100',
             'adaboosttreedepth30est10','adaboosttreedepth30est30','adaboosttreedepth30est50','adaboosttreedepth30est100','adaboosttreedepth40est10',
             'adaboosttreedepth40est30','adaboosttreedepth40est50','adaboosttreedepth40est100','adaboosttreedepth50est10','adaboosttreedepth50est30',
             'adaboosttreedepth50est50','adaboosttreedepth50est100'
          ]

for i in range(10):
    features_train, labels_train, features_test, labels_test = split_into_train_test_random(winedf)
    iteration_scores = []    
    for depth in [5, 10, 15, 20, 25, 30, 40, 50]:
        clf = tree.DecisionTreeClassifier(max_depth = depth)
        clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)
        acc = accuracy_score(pred, labels_test)
        print 'iteration', i, 'decision tree max depth =', depth, 'accuracy =', acc
        iteration_scores.append(acc)
   
    for neighbors in [2, 3, 5, 7, 10]:
        for wts in ['uniform', 'distance']: 
            clf = KNeighborsClassifier(n_neighbors=neighbors, weights=wts)
            clf.fit(features_train, labels_train)
            pred = clf.predict(features_test)
            acc = accuracy_score(pred, labels_test)
            print 'iteration', i, 'kneighbors k =', neighbors, 'weights = ', wts, 'accuracy =', acc
            iteration_scores.append(acc)

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
            pred = clf.predict(features_test)
            acc = accuracy_score(pred, labels_test)
            print 'iteration', i, 'adaboost tree depth =', depth, 'estimators =', estimators, 'accuracy =', acc
            iteration_scores.append(acc)
    #print len(iteration_scores)           
    all_iterations_scores.append(iteration_scores)
    #break
    
#print_list(all_iterations_scores, 'all scores')
scoresdf = pd.DataFrame(all_iterations_scores)

for i in range(len(scoresdf.columns)):
    l = scoresdf[i].values.tolist()
    print 'Model {}, Avg. accuracy = {}'.format(models[i], sum(l) / float(len(l)))
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
