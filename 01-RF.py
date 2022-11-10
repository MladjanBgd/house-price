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
#git push -u origin master

import numpy as np
import pandas as pd
import seaborn as sns


train_ds = pd.read_csv('./train.csv', sep=',')
test_ds = pd.read_csv('./test.csv', sep=',')

train_ds.head()
train_ds.info()

"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2918 entries, 0 to 2917
Data columns (total 78 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
  0   Id             1460 non-null   int64  
  1   MSSubClass     1460 non-null   int64  
  2   MSZoning       1460 non-null   object 
  3   LotFrontage    1201 non-null   float64
  4   LotArea        1460 non-null   int64  
  5   Street         1460 non-null   object 
  6   Alley          91 non-null     object 
  7   LotShape       1460 non-null   object 
  8   LandContour    1460 non-null   object 
  9   Utilities      1460 non-null   object 
  10  LotConfig      1460 non-null   object 
  11  LandSlope      1460 non-null   object 
  12  Neighborhood   1460 non-null   object 
  13  Condition1     1460 non-null   object 
  14  Condition2     1460 non-null   object 
  15  BldgType       1460 non-null   object 
  16  HouseStyle     1460 non-null   object 
  17  OverallQual    1460 non-null   int64  
  18  OverallCond    1460 non-null   int64  
  19  YearBuilt      1460 non-null   int64  
  20  YearRemodAdd   1460 non-null   int64  
  21  RoofStyle      1460 non-null   object 
  22  RoofMatl       1460 non-null   object 
  23  Exterior1st    1460 non-null   object 
  24  Exterior2nd    1460 non-null   object 
  25  MasVnrType     1452 non-null   object 
  26  MasVnrArea     1452 non-null   float64
  27  ExterQual      1460 non-null   object 
  28  ExterCond      1460 non-null   object 
  29  Foundation     1460 non-null   object 
  30  BsmtQual       1423 non-null   object 
  31  BsmtCond       1423 non-null   object 
  32  BsmtExposure   1422 non-null   object 
  33  BsmtFinType1   1423 non-null   object 
  34  BsmtFinSF1     1460 non-null   int64  
  35  BsmtFinType2   1422 non-null   object 
  36  BsmtFinSF2     1460 non-null   int64  
  37  BsmtUnfSF      1460 non-null   int64  
  38  TotalBsmtSF    1460 non-null   int64  
  39  Heating        1460 non-null   object 
  40  HeatingQC      1460 non-null   object 
  41  CentralAir     1460 non-null   object 
  42  Electrical     1459 non-null   object 
  43  1stFlrSF       1460 non-null   int64  
  44  2ndFlrSF       1460 non-null   int64  
  45  LowQualFinSF   1460 non-null   int64  
  46  GrLivArea      1460 non-null   int64  
  47  BsmtFullBath   1460 non-null   int64  
  48  BsmtHalfBath   1460 non-null   int64  
  49  FullBath       1460 non-null   int64  
  50  HalfBath       1460 non-null   int64  
  51  BedroomAbvGr   1460 non-null   int64  
  52  KitchenAbvGr   1460 non-null   int64  
  53  KitchenQual    1460 non-null   object 
  54  TotRmsAbvGrd   1460 non-null   int64  
  55  Functional     1460 non-null   object 
  56  Fireplaces     1460 non-null   int64  
  57  FireplaceQu    770 non-null    object 
  58  GarageType     1379 non-null   object 
  59  GarageYrBlt    1379 non-null   float64
  60  GarageFinish   1379 non-null   object 
  61  GarageCars     1460 non-null   int64  
  62  GarageArea     1460 non-null   int64  
  63  GarageQual     1379 non-null   object 
  64  GarageCond     1379 non-null   object 
  65  PavedDrive     1460 non-null   object 
  66  WoodDeckSF     1460 non-null   int64  
  67  OpenPorchSF    1460 non-null   int64  
  68  EnclosedPorch  1460 non-null   int64  
  69  3SsnPorch      1460 non-null   int64  
  70  ScreenPorch    1460 non-null   int64  
  71  PoolArea       1460 non-null   int64  
  72  PoolQC         7 non-null      object 
  73  Fence          281 non-null    object 
  74  MiscFeature    54 non-null     object 
  75  MiscVal        1460 non-null   int64  
  76  MoSold         1460 non-null   int64  
  77  YrSold         1460 non-null   int64  
  78  SaleType       1460 non-null   object 
  79  SaleCondition  1460 non-null   object 
  80  SalePrice      1460 non-null   int64  
dtypes: float64(11), int64(26), object(41)
memory usage: 1.7+ MB
"""

ds=pd.concat([train_ds,test_ds], ignore_index=True)

for col in ds:
    if ds[col].isna().sum() > 0: print('Col:',col,str(ds[col].isna().sum()))
    
"""
Col: MSZoning 4
Col: LotFrontage 486
Col: Alley 2721
Col: Utilities 2
Col: Exterior1st 1
Col: Exterior2nd 1
Col: MasVnrType 24
Col: MasVnrArea 23
Col: BsmtQual 81
Col: BsmtCond 82
Col: BsmtExposure 82
Col: BsmtFinType1 79
Col: BsmtFinSF1 1
Col: BsmtFinType2 80
Col: BsmtFinSF2 1
Col: BsmtUnfSF 1
Col: TotalBsmtSF 1
Col: Electrical 1
Col: BsmtFullBath 2
Col: BsmtHalfBath 2
Col: KitchenQual 1
Col: Functional 2
Col: FireplaceQu 1420
Col: GarageType 157
Col: GarageYrBlt 159
Col: GarageFinish 159
Col: GarageCars 1
Col: GarageArea 1
Col: GarageQual 159
Col: GarageCond 159
Col: PoolQC 2909
Col: Fence 2348
Col: MiscFeature 2814
Col: SaleType 1
Col: SalePrice 1459
"""

#fix NaN's

#remove for now FireplaceQu and MiscFeataure, too much NaN's...but theay are very important feature for price!?

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=1337, shuffle=True)

# rf = RandomForestRegressor()

# param_grid = {'n_estimators': [300,400,500],
#               "criterion" : ["squared_error", "absolute_error"],
#               "max_depth": [16,17,18,19,20,21],
#               "min_samples_split": [15, 16, 17,18],
#               "min_samples_leaf": [1,2,3,4]}

# ###GridSearchCV to slow because of num of dimensions -> using RandomizedSearchCV
# rs_cv = RandomizedSearchCV(estimator=rf, n_iter=10,
#                            param_distributions=param_grid, cv=5, n_jobs=-1,
#                            random_state=42, scoring='neg_mean_squared_error', verbose=10)
# res = rs_cv.fit(X_train, y_train)

# print(res.best_params_)
# print((-1 * res.best_score_) ** 0.5)


# # # # """
# # # # {'n_estimators': 400, 'min_samples_split': 17, 'min_samples_leaf': 1, 'max_depth': 19, 'criterion': 'squared_error'}
# # # # """
# joblib.dump(res.best_estimator_, 'rs_cv_m.pkl')



rf=RandomForestRegressor(n_estimators=50, criterion='squared_error', max_depth=12, min_samples_split=20, min_samples_leaf=2)
rf=rf.fit(X_train,y_train)

y_pred=rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = mse**.5
print(mse)
print(rmse)

"""
562 330 286.2862304
23713.504302110865
"""

y_res = ds_enc.loc[1460:2919, 'SalePrice']
X_res = ds_enc.loc[1460:2919, :]
X_res = X_res.drop('SalePrice', axis=1)

y_res=rf.predict(X_res)


res=pd.DataFrame(X_res.loc[:,'Id'])
res['SalePrice'] = y_res

save_res=res.to_csv('./sample_submission.csv', index=False)









    



