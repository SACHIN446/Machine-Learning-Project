import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score

df=pd.read_csv("car_dataset.csv",usecols=['SystemCodeNumber','Capacity','Occupancy'],nrows=2000)

#--------------------DATASET ANALYSIS--------------------------

print("\n=======================DATASET ANALYSIS=======================")

print("-----------------NAN Values----------------")
print(df.isna().sum(),"\n") 

print("-----------------value_count-----------------")
print(df.dtypes.value_counts,"\n") 

print("-------------DATASET info----------------------")
print(df.info(),"\n")

print("-------------DATASET describe----------------------")
print(df.describe())

X=df[['Capacity','Occupancy']]
Y=df['SystemCodeNumber']


#------------LABEL ECODER---------
le=LabelEncoder()
Y=le.fit_transform(Y)

xtrain,xtest, ytrain,ytest = train_test_split(X,Y,test_size=0.3)
#print(np.unique(Y)

#------------------------------Classifiers---------------------

print("\n====================ALL CLASSIFIER OUTPUT========================\n")
from sklearn.svm import SVC
clf1=SVC(C=1,kernel='rbf', degree=3, gamma='auto', class_weight=None, decision_function_shape='ovr')
clf1.fit(xtrain,ytrain)
clf1_pred=clf1.predict(xtest)
print("Accuracy Score of SVC:",accuracy_score(clf1_pred,ytest))
print("Correctly Classified/Total Sample ",accuracy_score(ytest,clf1_pred,normalize=False),"/",len(ytest),"\n")


from sklearn.svm import LinearSVC
clf2=LinearSVC(penalty='l1', loss='squared_hinge', C=1, multi_class='ovr', fit_intercept=False, class_weight=None, dual=False)
clf2.fit(xtrain,ytrain)
clf2_pred=clf2.predict(xtest)
print("Accuracy Score Of LinearSVC ",accuracy_score(ytest,clf2_pred))
print("Correctly Classified/Total Sample ",accuracy_score(ytest,clf2_pred,normalize=False),"/",len(ytest),"\n")


from sklearn.svm import NuSVC
clf3 = NuSVC(nu=0.5, kernel='rbf', degree=3, gamma='auto', decision_function_shape='ovr')
clf3.fit(xtrain,ytrain)
clf3_pred = clf3.predict(xtest)
print("Accuracy Score Of NuSVC ",accuracy_score(ytest,clf3_pred))
print("Correctly Classified/Total Sample ",accuracy_score(ytest,clf3_pred,normalize=False),"/",len(ytest),"\n")


from sklearn.tree import DecisionTreeClassifier
clf4 = DecisionTreeClassifier(criterion='gini', class_weight=None)
clf4.fit(xtrain,ytrain)
clf4_pred = clf4.predict(xtest)
print("Accuracy Score Of DecisionTreeClassifier ",accuracy_score(ytest,clf4_pred))
print("Correctly Classified/Total Sample ",accuracy_score(ytest,clf4_pred,normalize=False),"/",len(ytest),"\n")


from sklearn.neighbors import KNeighborsClassifier
clf6 = KNeighborsClassifier(n_neighbors=100, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski')
clf6.fit(xtrain,ytrain)
clf6_pred = clf6.predict(xtest)
print("Accuracy Score Of KNeighborsClassifier ",accuracy_score(ytest,clf6_pred))
print("Correctly Classified/Total Sample ",accuracy_score(ytest,clf6_pred,normalize=False),"/",len(ytest),"\n")


from sklearn.naive_bayes import GaussianNB
clf7 = GaussianNB(priors=None, var_smoothing=1e-09)
clf7.fit(xtrain,ytrain)
clf7_pred = clf7.predict(xtest)
print("Accuracy Score Of GaussianNB ",accuracy_score(ytest,clf7_pred))
print("Correctly Classified/Total Sample ",accuracy_score(ytest,clf7_pred,normalize=False),"/",len(ytest),"\n")

from sklearn.naive_bayes import MultinomialNB
clf8 = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None) 
clf8.fit(xtrain,ytrain)
clf8_pred = clf8.predict(xtest)
print("Accuracy Score Of MultinomialNB ",accuracy_score(ytest,clf8_pred))
print("Correctly Classified/Total Sample ",accuracy_score(ytest,clf8_pred,normalize=False),"/",len(ytest),"\n")


from sklearn.linear_model import LogisticRegression
clf9 = LogisticRegression(penalty='l2', dual=False, C=1, fit_intercept=True, class_weight=None, solver='lbfgs', multi_class='auto')
#solver = ‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’ (default=’lbfgs’)
# sag / saga are worst 
clf9.fit(xtrain,ytrain)
clf9_pred = clf9.predict(xtest)
print("Accuracy Score Of LogisticRegression ",accuracy_score(ytest,clf9_pred))
print("Correctly Classified/Total Sample ",accuracy_score(ytest,clf9_pred,normalize=False),"/",len(ytest),"\n")



#-------------------------Regressors-----------------------

print("\n==================Regression Models Output====================\n")

from sklearn.neighbors import RadiusNeighborsRegressor
r1=RadiusNeighborsRegressor(radius=1.0,weights='uniform',algorithm='auto',leaf_size=30,p=2,metric='minkowski')
r1.fit(xtrain,ytrain)
r1_pred=r1.predict(xtest)
print("Accuracy Score Of SVR ",accuracy_score(ytest,r1_pred))
print("Correctly Classified/Total Sample ",accuracy_score(ytest,r1_pred,normalize=False),"/",len(ytest),"\n")


from sklearn.tree import DecisionTreeRegressor
r4=DecisionTreeRegressor(criterion='mse',splitter='best',max_depth=None,min_samples_split=2,min_samples_leaf=1)
r4.fit(xtrain,ytrain)
r4_pred=r4.predict(xtest)
print("Accuracy Score Of SVR ",accuracy_score(ytest,r4_pred))
print("Correctly Classified/Total Sample ",accuracy_score(ytest,r4_pred,normalize=False),"/",len(ytest),"\n")


from sklearn.neighbors import KNeighborsRegressor
r5=KNeighborsRegressor(n_neighbors=10,weights='uniform',algorithm='auto',leaf_size=30,p=2,metric='minkowski')
r5.fit(xtrain,ytrain)
r5_pred=r5.predict(xtest)
print("Accuracy Score Of KNeighborsRegressor ",accuracy_score(ytest,r5_pred))
print("Correctly Classified/Total Sample ",accuracy_score(ytest,r5_pred,normalize=False),"/",len(ytest),"\n")


#------------------Clustering--------------
print("\n===================Clustering Models Output======================\n")
from sklearn.cluster import AgglomerativeClustering
cls1=AgglomerativeClustering(n_clusters=2,affinity='euclidean',distance_threshold=None)
cls1.fit(xtrain)
cls1_pred=cls1.labels_
print("AgglomerativeClustering Prediction",cls1_pred)

from sklearn.cluster import DBSCAN
cls2=DBSCAN(eps=1,min_samples=2,metric='euclidean',algorithm='auto',leaf_size=30,p=None)
cls2.fit(xtrain)
cls2_pred=cls2.labels_
print("DBSCAN Prediction",cls2_pred)

from sklearn.cluster import KMeans
cls3=KMeans(n_clusters=10,init='k-means++',n_init=10,algorithm='auto')
cls3.fit(xtrain)
cls3_pred=cls3.labels_
print("KMeans Prediction",cls3_pred)

from sklearn.cluster import Birch
cls4=Birch(threshold=0.5,branching_factor=50,n_clusters=3,compute_labels=True)
cls4.fit(xtrain)
cls4_pred = cls4.predict(xtrain)
print("Birch Prediction",cls4_pred)

#------------------Plotting Clusters-------------------

from matplotlib import pyplot as plt
plt.subplot(221)
plt.scatter(xtrain.iloc[:,0],xtrain.iloc[:,1],c=cls1_pred)
plt.title("AgglomerativeClustering")

plt.subplot(222)
plt.scatter(xtrain.iloc[:,0],xtrain.iloc[:,1],c=cls2_pred)
plt.title("DBSCAN")

plt.subplot(223)
plt.scatter(xtrain.iloc[:,0],xtrain.iloc[:,1],c=cls3_pred)
plt.title("KMeans")

plt.subplot(224)
plt.scatter(xtrain.iloc[:,0],xtrain.iloc[:,1],c=cls4_pred)
plt.title("FeatureAgglomeration")
plt.tight_layout(pad=1.5)
plt.show()






























