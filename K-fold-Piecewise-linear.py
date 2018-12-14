# -*- coding: utf-8 -*-
"""
CS229 project: Piecewise linear with unsupervised clusteringd

"""

from sklearn.model_selection import StratifiedKFold
import numpy as np
from itertools import cycle, islice
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression 
from sklearn import linear_model
from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import cross_val_score
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')
import os
from scipy.optimize import curve_fit
import xgboost as xgb
from itertools import combinations_with_replacement

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def manage_outliers(a,b):
    a[a>np.mean(a)+6*np.std(b)]=np.max(b)
    a[a<np.mean(a)-6*np.std(b)]=np.min(b)
    return a

path = r"C:\Users\MGadre153503\OneDrive - Applied Materials\Documents\CS229-Project\superconduct"

#path = r"C:\Users\vshah143959\Applied Materials\Milind Gadre - CS229-Project\superconduct"

os.chdir(path)
filename = r"\train.csv"
df = pd.read_csv(path+filename)
n = np.int(df.shape[0]*0.8)

    
X = df.iloc[:,:-1]
scaled_X = StandardScaler().fit_transform(X.values)
X = pd.DataFrame(scaled_X, columns=X.columns)
y = df.iloc[:,-1]
#1 extract test train
X_train_dev, X_test, y_train_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#Decisiontree based piecewise linear regression (LMT)
from sklearn import tree
rs = ShuffleSplit(n_splits=5, test_size=.25, random_state=0)
rs.get_n_splits(X_train_dev)
fold = 0
df_Rsq_DT = pd.DataFrame(columns=['Fold','Max depth','Leaf','MSE on train','MSE on dev','Rsq on train','Rsq on dev'])
for train_ind, test_ind in rs.split(X_train_dev):
    fold += 1
    X_train = X_train_dev.iloc[train_ind,:]
    X_dev = X_train_dev.iloc[test_ind,:]
    y_train = y_train_dev.iloc[train_ind]
    y_dev = y_train_dev.iloc[test_ind]
    y_train_Tc = y_train
    y_dev_Tc = y_dev
# k-means
    for max_depth in range(1,9):
        clf = tree.DecisionTreeRegressor(max_depth = max_depth,min_samples_leaf = 250)
        clf.fit(X_train,y_train)
        X_train_leavesID = clf.apply(X_train)
        X_dev_leavesID = clf.apply(X_dev)
        for leafID in np.unique(X_train_leavesID):
            X_train_i = X_train.loc[X_train_leavesID==leafID,:]
            y_train_i = y_train_Tc.loc[X_train_leavesID==leafID]
            X_dev_i = X_dev.loc[X_dev_leavesID==leafID,:]
            y_dev_i = y_dev_Tc.loc[X_dev_leavesID==leafID]

#            lr = linear_model.LinearRegression()
            lrRANSAC = linear_model.RANSACRegressor(random_state=0,min_samples=np.int(X_train_i.shape[0]*0.9))
            lrRANSAC.fit(X_train_i,y_train_i)
#            lr.fit(X_train_i,y_train_i)
            y_train_pred = lrRANSAC.predict(X_train_i)
            MSE_train = np.sum((y_train_i.values-y_train_pred)**2)/y_train_i.shape[0]
            Rsq_train = lrRANSAC.score(X_train_i,y_train_i)

            print(max_depth,leafID,MSE_train,Rsq_train,X_train_i.shape[0])
            y_dev_pred = lrRANSAC.predict(X_dev_i)
            MSE_dev = np.sum((y_dev_i.values-y_dev_pred)**2)/y_dev_i.shape[0]
            Rsq_dev = lrRANSAC.score(X_dev_i,y_dev_i)

#            classifierscore = lr.score(X_dev_i,y_dev_i)
#            y_dev_pred = lr.predict(X_dev_i)
#            y_dev_pred = manage_outliers(y_dev_pred,y_dev_i)
#            MSE_dev = np.sum((y_dev_i.values-y_dev_pred)**2)/y_dev_i.shape[0]
#            rss =  (y_dev_i.values-y_dev_pred)**2
#            tss =  (y_dev_i.values-np.mean(y_dev_i.values))**2
#            Rsq_dev = 1 - rss/tss
            df_Rsq_DT = df_Rsq_DT.append(pd.DataFrame(np.reshape([fold,max_depth,leafID,MSE_train,MSE_dev,Rsq_train,Rsq_dev],(1,7)),columns=df_Rsq_DT.columns))
sns.set();
sns.boxplot(x = df_Rsq_DT['Max depth'], y=df_Rsq_DT['Rsq on dev'],palette="Blues")
plt.ylabel('Rsq on dev dataset',fontsize=16)
plt.title('R^2 distribution of piecewise linear fits on dev',fontsize=18)
plt.ylim(0,1.1)
plt.suptitle("")
plt.show();plt.close()
sns.boxplot(x = df_Rsq_DT['Max depth'], y=df_Rsq_DT['Rsq on train'],palette="Blues")
plt.ylabel('Rsq on train dataset',fontsize=18)
plt.title('R^2 distribution of piecewise linear fits on train',fontsize=16)
plt.ylim(0,1.1)
plt.suptitle("")
plt.show();plt.close()

#k-means
rs = ShuffleSplit(n_splits=5, test_size=.25, random_state=0)
rs.get_n_splits(X_train_dev)
fold = 0
df_Rsq_kM = pd.DataFrame(columns=['fold','Number of clusters','Cluster','MSE on train','MSE on dev','Rsq on train','Rsq on dev'])
for train_ind, test_ind in rs.split(X_train_dev):
    fold += 1
    X_train = X_train_dev.iloc[train_ind,:]
    X_dev = X_train_dev.iloc[test_ind,:]
    y_train = y_train_dev.iloc[train_ind]
    y_dev = y_train_dev.iloc[test_ind]
# k-means
    for k_clusters in range(1,6):
        kmeans = KMeans(n_clusters=k_clusters, random_state=0).fit(X_train)
        Xdev_cluster_id = kmeans.predict(X_dev)
        for i in range(kmeans.n_clusters):
            X_train_i = X_train.loc[kmeans.labels_== i,:]
            y_train_i = y_train.loc[kmeans.labels_ == i]
            X_dev_i = X_dev.loc[Xdev_cluster_id == i,:]
            y_dev_i = y_dev.loc[Xdev_cluster_id == i]

            lrRANSAC = linear_model.RANSACRegressor(random_state=0,min_samples=np.int(X_train_i.shape[0]*0.9))
            lrRANSAC.fit(X_train_i,y_train_i)
            y_train_pred = lrRANSAC.predict(X_train_i)
            Rsq_train = lrRANSAC.score(X_train_i,y_train_i)
            MSE_train = np.sum((y_train_i.values-y_train_pred)**2)/y_train_i.shape[0]

            y_dev_pred = lrRANSAC.predict(X_dev_i)
            Rsq_dev = lrRANSAC.score(X_dev_i,y_dev_i)
            MSE_dev = np.sum((y_dev_i.values-y_dev_pred)**2)/y_dev_i.shape[0]
            df_Rsq_kM = df_Rsq_kM.append(pd.DataFrame(np.reshape([fold,k_clusters,i+1,MSE_train,MSE_dev,Rsq_train,Rsq_dev],(1,7)),columns=df_Rsq_kM.columns))
            print(fold,k_clusters,i+1,MSE_train,MSE_dev,Rsq_train,Rsq_dev)
sns.set();
sns.boxplot(x = df_Rsq_kM['Number of clusters'], y=df_Rsq_kM['Rsq on dev'],palette="Blues")
plt.ylabel('Rsq on dev dataset',fontsize=16)
plt.title('R^2 distribution of piecewise linear fits on dev',fontsize=18)
plt.ylim(0,1.1)
plt.suptitle("")
plt.show();plt.close()
sns.boxplot(x = df_Rsq_kM['Number of clusters'], y=df_Rsq_kM['Rsq on train'],palette="Blues")
plt.ylabel('Rsq on train dataset',fontsize=18)
plt.title('R^2 distribution of piecewise linear fits on train',fontsize=16)
plt.ylim(0,1.1)
plt.suptitle("")
plt.show();plt.close()
    
    