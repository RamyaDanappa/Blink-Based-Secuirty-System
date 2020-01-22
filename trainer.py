import pickle
import pandas as pd
import numpy as np
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

def svmtrnr(usn,trd):
    data = pickle.load(  open(usn +"_"+trd+".pickle","rb") )
    redo = 0
    dff = []
    for i in range(len(data)):
        if i >= 7 and i < len(data)-7:
            aear = (data[i][1]+data[i-7][1]+data[i-6][1]+data[i-5][1]+data[i-4][1]+ data[i-3][1]+ data[i-2][1]+ data[i-1][1]+data[i+7][1]+data[i+6][1]+data[i+5][1]+data[i+4][1]+ data[i+3][1]+ data[i+2][1]+ data[i+1][1]+ data[i][-1])/15.0
            
            dff.append([data[i][1],data[i-7][1],data[i-6][1],data[i-5][1],data[i-4][1], data[i-3][1], data[i-2][1], data[i-1][1], data[i+7][1],data[i+6][1],data[i+5][1],data[i+4][1], data[i+3][1], data[i+2][1], data[i+1][1], aear, data[i][-1]])
     
    d = pd.DataFrame(dff, columns=[ 'ear', 'ear-7', 'ear-6', 'ear-5', 'ear-4', 'ear-3', 'ear-2','ear-1', 'ear+7', 'ear+6', 'ear+5', 'ear+4', 'ear+3', 'ear+2', 'ear+1','aear','bl'])

    Y = d['bl']
    X = d.drop(columns=['bl'])

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=0)

    clf = svm.SVC(kernel='linear', C=2).fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Accuracy:",metrics.accuracy_score(y_test, y_pred)," Precision:", metrics.precision_score(y_test, y_pred)," Recall:",metrics.recall_score(y_test, y_pred))

    name = usn + '_' + trd + '_SVM.pickle'

    pre = metrics.precision_score(y_test, y_pred)
    if pre < 0.88 or pre > 0.99:
        redo = 1
    else:
        pickle.dump(clf,open(name,'wb'))
    return redo 
    

   