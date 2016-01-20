import pandas as pd
import random
from sklearn import tree

def split_into_train_test(df):
    s=set(range(len(df)))
    train_set = df
    test_set = []
    to_be_removed = []
    print len(df)
    
    while len(test_set) <= 25:
        y = random.choice(list(s))
        s.remove(y)
        to_be_removed.append(y)
        test_set.append(list(df.iloc[y]))
    df.drop(df.index[to_be_removed], inplace=True)
    for row in test_set:
        print row
    print len(to_be_removed)
    test_df = pd.DataFrame(test_set)
    print test_df
    
    df.drop([0], inplace=True)
    
    test_df.drop([0], inplace=True)
    #print df
    #print test_df
    print len(df)
    print len(test_df)
    features_train = df.values.tolist()
    features_test = test_df.values.tolist()
    labels_train = list(df[0])
    labels_test = list(test_df[0])
    print len(features_train), len(labels_train), len(features_test), len(labels_test)
    return features_train, labels_train, features_test, labels_test

winedf = pd.read_csv('wine_data.csv', header=None)
features_train, labels_train, features_test, labels_test = split_into_train_test(winedf)

clf50 = tree.DecisionTreeClassifier()
clf50.fit(features_train, labels_train)
pred50 = clf50.predict(features_test)


from sklearn.metrics import accuracy_score
#acc = accuracy_score(pred, labels_test)
#print acc


acc_min_samples_split_50 = accuracy_score(pred50, labels_test)

print acc_min_samples_split_50
