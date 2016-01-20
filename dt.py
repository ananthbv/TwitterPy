import pandas as pd
import random
from sklearn import tree
#from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

label_column = 0

def print_list(lst, cmt):
    print cmt, '='
    for row in lst:
        print row

def split_into_train_test(df):
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

    print 'len(features_train), len(labels_train), len(features_test), len(labels_test) = ', len(features_train), len(labels_train), len(features_test), len(labels_test)
    #print_list(features_train, "features_train")
	#print_list(labels_train, "features_train")
    #print_list(features_test, "features_train")
    #print_list(labels_test, "features_train")
    return features_train, labels_train, features_test, labels_test

winedf = pd.read_csv('wine_data.csv', header=None)

features_train, labels_train, features_test, labels_test = split_into_train_test(winedf)

#clf = tree.DecisionTreeClassifier(max_depth = 25)
#clf = KNeighborsClassifier(n_neighbors=51)
#clf = SVC(kernel="linear", C=0.025)
#clf = SVC(gamma=2, C=0.025)
clf = SVC()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)


#acc = accuracy_score(pred, labels_test)
#print acc

#acc_min_samples_split_50 = accuracy_score(pred50, labels_test)
acc = accuracy_score(pred, labels_test)

print acc
