x = {0: 1, 1: 2, 3: 11, 11: 19, 4: 7, 7: 10, 10: 29, 6: 8, 8: 9, 30: 31, 31: 32}

def loopThru(k, v):
    x[k] = -1
    print v
    if v in x.keys():
        return loopThru(v, x[v])
    else:
        return False

for k, v in x.iteritems():
    print 'checking k =', k
    loopThru(k, v)
    print ''




