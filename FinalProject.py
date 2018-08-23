import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import  CountVectorizer
nltk.download('stopwords')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier

#6221 features in feature vector


dataset = pd.read_csv('C:/Users/hmeji/Desktop/315/Project/spam.csv',encoding='latin-1')
dataset = dataset.drop(['Unnamed: 2' , 'Unnamed: 3', 'Unnamed: 4'],axis=1)
dataset = dataset.rename(columns={'v1':'Target','v2':'SMS'})
#print(dataset.head())

ps = PorterStemmer()
corpus = []

for i in range(len(dataset['SMS'])):
 clean_sms = re.sub('[^a-zA-Z]',' ',dataset['SMS'][i])
 clean_sms = clean_sms.lower()
 clean_sms = clean_sms.split()
 clean_sms = [ps.stem(word) for word in clean_sms if not word in set(stopwords.words('english'))]
 clean_sms = ' '.join(clean_sms)
 corpus.append(clean_sms)

cv = CountVectorizer()

X = cv.fit_transform(corpus).toarray()

y = dataset.iloc[:,0].values

for i in range(len(y)):
    if y[i] == 'ham':
        y[i] = 1
    else:
        y[i] = 0

y = y.astype(np.int64)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)

print("Perceptron: test results")
L = [1,5,10,20,50]
N = [0.1, 0.5, 1.0,5.0,10.0]
for j in range(0, 5):
    for i in range(0, 5):
        ppn = Perceptron(n_iter=L[j], eta0=N[i], random_state=0)
        ppn.fit(X_train_std, y_train)
        y_pred = ppn.predict(X_test_std)
        print('n_iter: ', L[j], end=" ")
        print('learn_rate: ', N[i], end=" ")
        print('| Accuracy: %.2f' % accuracy_score(y_test, y_pred))
"""
X_train_std = sc.transform(X_train)
print("Random Forest: test results")
L = [1,5,10,20,50,100,200]
for i in range(0, 7):
    rf = RandomForestClassifier(n_estimators=L[i])
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print('n_trees: ', L[i], end=" ")
    print('| Accuracy: %.2f' % accuracy_score(y_test, y_pred))

X_train_std = sc.transform(X_train)
print("SVM: test results")
clf = svm.SVC()
clf.fit(X, y)
y_pred = clf.predict(X_test)
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

X_train_std = sc.transform(X_train)
print("ADA Boost: test results")
L = [1,5,10,20,50,100,200]
for i in range(0, 7):
    adaClf = AdaBoostClassifier(n_estimators=L[i])
    adaClf.fit(X, y)
    y_pred = adaClf.predict(X_test)
    print('n_estimators: ', L[i], end=" ")
    print('| Accuracy: %.2f' % accuracy_score(y_test, y_pred))
"""


"""
ppn = Perceptron(n_iter=1, eta0=0.1, random_state=0)

# Train the perceptron
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)

rf = RandomForestClassifier(n_estimators = 200)
rf.fit(X_train,y_train)
y_pred2 = rf.predict(X_test)

clf = svm.SVC()
clf.fit(X, y)
y_pred3 = clf.predict(X_test)

adaClf = AdaBoostClassifier(n_estimators=100)
adaClf.fit(X,y)
y_pred4 = clf.predict(X_test)

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred2))
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred3))
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred4))
"""