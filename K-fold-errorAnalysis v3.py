# -*- coding: utf-8 -*-
"""
CS229 project analysis with given features
"""

from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPRegressor
import numpy as np
from itertools import cycle, islice
import pandas as pd
import matplotlib.pyplot as plt
#matplotlib.rcParams
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression 
from sklearn import linear_model
from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import cross_val_score
import xgboost
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')
import os
from scipy.optimize import curve_fit


path = r"C:\Users\MGadre153503\OneDrive - Applied Materials\Documents\CS229-Project\superconduct"

path = r"C:\Users\vshah143959\Applied Materials\Milind Gadre - CS229-Project\superconduct"

os.chdir(path)
os.chdir(path+'\Figures')
os.chdir(path)
filename = r"\train.csv"
df = pd.read_csv(path+filename)

df = df.iloc[:,:]
X = df.iloc[:,:-1]
scaled_X = StandardScaler().fit_transform(X.values)
X = pd.DataFrame(scaled_X, columns=X.columns)
y = df.iloc[:,-1]
#1 extract test train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
n = X_train.shape[0]
sizearray = np.array(list(map(lambda x: int(x), np.linspace(1000,n,7))))
sizearray = sizearray[::-1]

cv = ShuffleSplit(X_train.shape[0], n_iter=5, test_size=0.2, random_state=0)

#instantiate estimators
default_base = {'random_state': 0,
                'n_estimators': 10,
                'fit_intercept':True, 
                'solver':'lsqr', 
                'tol':0.001,
                'min_samples_leaf':100,
                'max_iter':500,
                'hidden_layer_sizes':(2,2,2,2)
                }
params = default_base.copy()
lrRidge = linear_model.Ridge(fit_intercept=params['fit_intercept'],
                             solver=params['solver'], tol=params['tol'])
lrLasso = linear_model.Lasso(fit_intercept=params['fit_intercept'],
                             tol=params['tol'])
regrDT = DecisionTreeRegressor(min_samples_leaf=params['min_samples_leaf'])
regrRF = RandomForestRegressor(random_state=params['random_state'], n_estimators=params['n_estimators'], min_samples_leaf=params['min_samples_leaf']) 
regMLP = MLPRegressor(hidden_layer_sizes=params['hidden_layer_sizes'], max_iter=params['max_iter'])
xgb = xgboost.XGBRegressor(booster='gbtree', subsample=0.75,min_samples_leaf=params['min_samples_leaf'],
                       colsample_bytree=1, n_estimators=200,learning_rate=0.07,max_depth=9)
regression_algorithms = (
        ('OLS + Ridge Loss',lrRidge,'alpha',np.array([0,0.5,1., 2., 3., 4., 5.,6,7])),
        ('OLS + Lasso Loss', lrLasso,'alpha',np.array([0,0.5,1., 2., 3., 4., 5.])),
        ('DecisionTree Regression',regrDT,'max_depth',np.linspace(2, 16, 8)),
        ('RandomForest Regression',regrRF,'max_depth',np.linspace(2, 16, 8)),
        ('Neural Net', regMLP, 'activation', np.array(['relu','tanh','logistic'])),
        ('XGboost', xgb, 'gamma', np.array([0.1,0.2,0.5])),#,'learning_rate', np.array([0.05,0.07])),
        )
#instantiation done
for name, estimator, keyword, hyperparams in regression_algorithms: 
    kwargs = {keyword: hyperparams}
    classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=kwargs)
    classifier.fit(X_train, y_train)
    classifierscore = np.round(classifier.score(X_test, y_test),decimals=2)
    trainclassifierscore = np.round(classifier.score(X_train, y_train),decimals=2)
    print(name, 'score: ',classifier.score(X_test, y_test), 'train',classifier.score(X_train, y_train))
    print('best parameters found: ', classifier.best_params_)
    if type(classifier.best_params_[keyword]) == np.str_:
        bestparam = classifier.best_params_[keyword]
    else:
        bestparam = int(classifier.best_params_[keyword])

#plot scores vs params'''
    plt.figure(figsize=(5,5))
    scores = [x[1] for x in classifier.grid_scores_]
    plt.scatter(hyperparams,np.round(scores,decimals=2))#np.round(1-trainclassifierscore,decimals=2)
#    plt.ylim(0,1)
    plt.xlabel(keyword,fontsize=16)
    plt.ylabel(r'$\mathregular{R^2}$',fontsize=16)
    plt.title('Hyper parameter tuning',fontsize=18)
    if name=='OLS + Ridge Loss':
        plt.ylim(0,1)
    else:
        plt.ylim(0,1)
    plt.text(.99, .01, 'best '+keyword+'='+str(bestparam),
             transform=plt.gca().transAxes, size=15,
             horizontalalignment='right')
    plt.tight_layout(0.5)
    plt.savefig(path+'\Figures\\'+name+'_Hyperparam.png')
    plt.show()
    plt.close()
    
#plot y_pred vs y_actual'''
    plt.figure(figsize=(5,5))
    plt.scatter(y_test,classifier.predict(X_test))
    plt.title(name,fontsize=18)
    plt.xlabel('Actual Temperature(K)',fontsize=16)
    plt.ylabel('Predicted Temperature(K)',fontsize=16)
    plt.text(.99, .01, keyword+'='+str(bestparam)+ r'$\mathregular{, R^2 =}$'+str(classifierscore),
             transform=plt.gca().transAxes, size=15,
             horizontalalignment='right')
    plt.tight_layout(0.5)
    plt.savefig(path+'\Figures\\'+name+'_pred_vs_act.png')
    plt.show()
    plt.close()
        
        
# Error analysis w.r.t. feature size
m = X_train.shape[1]
pcacols = ['pca_'+str(val) for loc,val in enumerate(np.arange(1,m+1,1))]
pca=PCA()
X_train_PCA = pd.DataFrame(pca.fit(X_train).transform(X_train),columns=pcacols)
X_test_PCA = pd.DataFrame(pca.fit(X_train).transform(X_test),columns=pcacols)
sizearray = np.array(list(map(lambda x: int(x), np.linspace(1,m,7))))
sizearray = sizearray[::-1]

for name, estimator, keyword, hyperparams in regression_algorithms: 
    for s in sizearray:
        X_i = X_train_PCA.iloc[:,:s]
        X_test_i = X_test_PCA.iloc[:,:s]
        kwargs = {keyword: hyperparams}
        classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=kwargs)
        classifier.fit(X_i, y_train)
        classifierscore = np.round(classifier.score(X_test_i, y_test),decimals=2)
        trainclassifierscore = np.round(classifier.score(X_i, y_train),decimals=2)
        print(name, s, 'score: ',classifier.score(X_test_i, y_test))
        print('best parameters found: ', classifier.best_params_)
        if type(classifier.best_params_[keyword]) == np.str_:
            bestparam = classifier.best_params_[keyword]
        else:
            bestparam = int(classifier.best_params_[keyword])
        plt.plot(s,np.round(1-classifierscore,decimals=2),'ro',s,np.round(1-trainclassifierscore,decimals=2),'g^')
    plt.xlabel('# of PCA components',fontsize=16)
    plt.ylabel('error ' + r'$\mathregular{(1-R^2)}$',fontsize=16)
    plt.title('Error analysis for '+name)
    plt.ylim(0,1)
    plt.tight_layout(0.5)
    plt.savefig(path+'\Figures\\'+name+'_feature_size.png')
    plt.show()
    plt.close()

""" for plotting with # of training examples"""
sizearray = np.array(list(map(lambda x: int(x), np.linspace(1000,n,7))))
sizearray = sizearray[::-1]
for name, estimator, keyword, hyperparams in regression_algorithms: 
    plt.figure()
    for s in sizearray:
        X_train_i = X_train.iloc[:s,:]
        y_train_i = y_train.iloc[:s]
        cv = ShuffleSplit(X_train_i.shape[0], n_iter=5, test_size=0.2, random_state=0)
        kwargs = {keyword: hyperparams}
        classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=kwargs)
        classifier.fit(X_train_i, y_train_i)
        classifierscore = np.round(classifier.score(X_test, y_test),decimals=2)
        trainclassifierscore = np.round(classifier.score(X_train_i, y_train_i),decimals=2)
        print(name, s, 'score: ',classifier.score(X_test, y_test))
        print('best parameters found: ', classifier.best_params_)
        if type(classifier.best_params_[keyword]) == np.str_:
            bestparam = classifier.best_params_[keyword]
        else:
            bestparam = int(classifier.best_params_[keyword])
        plt.plot(s,np.round(1-classifierscore,decimals=2),'ro',s,np.round(1-trainclassifierscore,decimals=2),'g^')
    plt.xlabel('# of training examples',fontsize=16)
    plt.ylabel('error ' + r'$\mathregular{(1-R^2)}$',fontsize=16)
    plt.title('Error analysis for '+name)
    plt.tight_layout(0.5)
    plt.ylim(0,1)
    plt.savefig(path+'\Figures\\'+name+'_sample_size.png')
    plt.show()
    plt.close()

# Plotting the importance of various features
xgboost.plot_importance(classifier.best_estimator_)
z = classifier.best_estimator_.feature_importances_
y = np.argsort(z)
x = y[::-1]
df.columns[x]
df.columns[x[0:8]]
plt.figure(figsize=(7,5))
plt.barh( list(map(lambda x: x.replace("_", " "), df.columns[x[0:8]])),z[x[0:8]])
#plt.xlabel('Important features',fontsize=22)
plt.ylabel('Weight',fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout(0.5)
plt.savefig(path+'\Figures\\'+name+'important_features.png')