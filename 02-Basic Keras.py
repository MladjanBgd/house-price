# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 13:03:28 2022

@author: mladjan.jovanovic
"""
#fresh new repo
#git init
#git add .
#git commit -m "..."
#git remote add origin https://github.com/MladjanBgd/house-price
#git push -u origin main

#https://www.kaggle.com/code/tomasmantero/predicting-house-prices-keras-ann
#https://www.kaggle.com/code/ironfrown/deep-learning-house-price-prediction-keras



import numpy as np
import pandas as pd
import seaborn as sns


train_ds = pd.read_csv('./train.csv', sep=',')
test_ds = pd.read_csv('./test.csv', sep=',')

# train_ds.head()
# train_ds.info()

ds=pd.concat([train_ds,test_ds], ignore_index=True)

# for col in ds:
#     if ds[col].isna().sum() > 0: print('Col:',col,str(ds[col].isna().sum()))
    

ds.drop(['Id'], axis=1, inplace=True)
ds = ds.drop(['FireplaceQu','MiscFeature'], axis=1)

#for encoder
objCol=[]
numCol=[]

for col in ds:
    if ds[col].dtype=="O":
        objCol.append(col)
    else:
        numCol.append(col)
    ds[col]=ds[col].replace(np.nan,ds[col].mode()[0])
    
ds_enc=pd.get_dummies(ds, columns=objCol)

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

y = ds_enc.loc[0:1460, 'SalePrice']
X = ds_enc.loc[0:1460, :]
X = X.drop('SalePrice', axis=1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# define base model
def baseline_model():
 	# create model
 	model = Sequential()
 	model.add(Dense(279, input_shape=(279,), kernel_initializer='normal', activation='relu'))
 	model.add(Dense(100, kernel_initializer='normal'))
 	model.add(Dense(1, kernel_initializer='normal'))
 	# Compile model
 	model.compile(loss='mean_squared_error', optimizer='adam')
 	return model
 
# evaluate model
#estimator = KerasRegressor(model=baseline_model, epochs=100, batch_size=5, verbose=1)
estimator = KerasRegressor(model=baseline_model, epochs=100, batch_size=5, verbose=1)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, X, y, cv=kfold, scoring='neg_mean_squared_error')
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# rf=RandomForestRegressor(n_estimators=50, criterion='squared_error', max_depth=12, min_samples_split=20, min_samples_leaf=2)
# rf=rf.fit(X_train,y_train)

# y_pred=rf.predict(X_test)

# mse = mean_squared_error(y_test, y_pred)
# rmse = mse**.5
# print(mse)
# print(rmse)


# y_res = ds_enc.loc[1460:2919, 'SalePrice']
# X_res = ds_enc.loc[1460:2919, :]
# X_res = X_res.drop('SalePrice', axis=1)

# y_res=rf.predict(X_res)

# res=pd.DataFrame(X_res.loc[:,'Id'])
# res['SalePrice'] = y_res

# save_res=res.to_csv('./sample_submission.csv', index=False)









    



