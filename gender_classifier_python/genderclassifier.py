from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# datasets
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],[190, 90, 47], [175, 64, 39],[177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]] 
Y = ['male', 'male', 'female', 'female', 'male', 'male','female', 'female','female', 'male', 'male']

#classifiers
dtc = tree.DecisionTreeClassifier()
svmc = SVC()
perC = Perceptron()
KNNc = KNeighborsClassifier()

#model training
dtc = dtc.fit(X,Y)
svmc = svmc.fit(X,Y)
perC = perC.fit(X,Y)
KNNc = KNNc.fit(X,Y)

# prediction
prediction1 = dtc.predict(X)
prediction2 = svmc.predict(X)
prediction3 = perC.predict(X)
prediction4 = KNNc.predict(X)

#accuracy
acc_dtc = accuracy_score(Y,prediction1)
print("dtc",acc_dtc)
acc_svmc = accuracy_score(Y,prediction2)
print("svmc",acc_svmc)
acc_perC = accuracy_score(Y,prediction3)
print("perC",acc_perC)
acc_knnc = accuracy_score(Y,prediction4)
print("knnc",acc_knnc)


# print best result
mval = max(acc_dtc,acc_svmc,acc_perC,acc_knnc)
acc = {acc_dtc : "decsion tree",acc_svmc : "svm",acc_perC : "Perceptron",acc_knnc :"KNN"}
print ("best gender classifier is {}".format(acc[mval]))



