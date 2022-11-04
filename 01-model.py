# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 13:03:28 2022

@author: mladjan.jovanovic
"""

import numpy as np
import pandas as pd
import seaborn as sns


train_ds = pd.read_csv('./test.csv')
test_ds = pd.read_csv('./test.csv')

train_ds.head()
train_ds.info()

"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2918 entries, 0 to 2917
Data columns (total 78 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   Id             2918 non-null   int64  
 1   MSSubClass     2918 non-null   int64  
 2   MSZoning       2918 non-null   object 
 3   LotFrontage    2918 non-null   float64
 4   LotArea        2918 non-null   int64  
 5   Street         2918 non-null   object 
 6   Alley          2918 non-null   object 
 7   LotShape       2918 non-null   object 
 8   LandContour    2918 non-null   object 
 9   Utilities      2918 non-null   object 
 10  LotConfig      2918 non-null   object 
 11  LandSlope      2918 non-null   object 
 12  Neighborhood   2918 non-null   object 
 13  Condition1     2918 non-null   object 
 14  Condition2     2918 non-null   object 
 15  BldgType       2918 non-null   object 
 16  HouseStyle     2918 non-null   object 
 17  OverallQual    2918 non-null   int64  
 18  OverallCond    2918 non-null   int64  
 19  YearBuilt      2918 non-null   int64  
 20  YearRemodAdd   2918 non-null   int64  
 21  RoofStyle      2918 non-null   object 
 22  RoofMatl       2918 non-null   object 
 23  Exterior1st    2918 non-null   object 
 24  Exterior2nd    2918 non-null   object 
 25  MasVnrType     2918 non-null   object 
 26  MasVnrArea     2918 non-null   float64
 27  ExterQual      2918 non-null   object 
 28  ExterCond      2918 non-null   object 
 29  Foundation     2918 non-null   object 
 30  BsmtQual       2918 non-null   object 
 31  BsmtCond       2918 non-null   object 
 32  BsmtExposure   2918 non-null   object 
 33  BsmtFinType1   2918 non-null   object 
 34  BsmtFinSF1     2918 non-null   float64
 35  BsmtFinType2   2918 non-null   object 
 36  BsmtFinSF2     2918 non-null   float64
 37  BsmtUnfSF      2918 non-null   float64
 38  TotalBsmtSF    2918 non-null   float64
 39  Heating        2918 non-null   object 
 40  HeatingQC      2918 non-null   object 
 41  CentralAir     2918 non-null   object 
 42  Electrical     2918 non-null   object 
 43  1stFlrSF       2918 non-null   int64  
 44  2ndFlrSF       2918 non-null   int64  
 45  LowQualFinSF   2918 non-null   int64  
 46  GrLivArea      2918 non-null   int64  
 47  BsmtFullBath   2918 non-null   float64
 48  BsmtHalfBath   2918 non-null   float64
 49  FullBath       2918 non-null   int64  
 50  HalfBath       2918 non-null   int64  
 51  BedroomAbvGr   2918 non-null   int64  
 52  KitchenAbvGr   2918 non-null   int64  
 53  KitchenQual    2918 non-null   object 
 54  TotRmsAbvGrd   2918 non-null   int64  
 55  Functional     2918 non-null   object 
 56  Fireplaces     2918 non-null   int64  
 57  GarageType     2918 non-null   object 
 58  GarageYrBlt    2918 non-null   float64
 59  GarageFinish   2918 non-null   object 
 60  GarageCars     2918 non-null   float64
 61  GarageArea     2918 non-null   float64
 62  GarageQual     2918 non-null   object 
 63  GarageCond     2918 non-null   object 
 64  PavedDrive     2918 non-null   object 
 65  WoodDeckSF     2918 non-null   int64  
 66  OpenPorchSF    2918 non-null   int64  
 67  EnclosedPorch  2918 non-null   int64  
 68  3SsnPorch      2918 non-null   int64  
 69  ScreenPorch    2918 non-null   int64  
 70  PoolArea       2918 non-null   int64  
 71  PoolQC         2918 non-null   object 
 72  Fence          2918 non-null   object 
 73  MiscVal        2918 non-null   int64  
 74  MoSold         2918 non-null   int64  
 75  YrSold         2918 non-null   int64  
 76  SaleType       2918 non-null   object 
 77  SaleCondition  2918 non-null   object 
dtypes: float64(11), int64(26), object(41)
memory usage: 1.7+ MB
"""

ds=pd.concat([train_ds,test_ds], ignore_index=True)

for col in ds:
    print('Col:',col,str(ds[col].isna().sum()))
    
"""
Col: Id 0
Col: MSSubClass 0
Col: MSZoning 8
Col: LotFrontage 454
Col: LotArea 0
Col: Street 0
Col: Alley 2704
Col: LotShape 0
Col: LandContour 0
Col: Utilities 4
Col: LotConfig 0
Col: LandSlope 0
Col: Neighborhood 0
Col: Condition1 0
Col: Condition2 0
Col: BldgType 0
Col: HouseStyle 0
Col: OverallQual 0
Col: OverallCond 0
Col: YearBuilt 0
Col: YearRemodAdd 0
Col: RoofStyle 0
Col: RoofMatl 0
Col: Exterior1st 2
Col: Exterior2nd 2
Col: MasVnrType 32
Col: MasVnrArea 30
Col: ExterQual 0
Col: ExterCond 0
Col: Foundation 0
Col: BsmtQual 88
Col: BsmtCond 90
Col: BsmtExposure 88
Col: BsmtFinType1 84
Col: BsmtFinSF1 2
Col: BsmtFinType2 84
Col: BsmtFinSF2 2
Col: BsmtUnfSF 2
Col: TotalBsmtSF 2
Col: Heating 0
Col: HeatingQC 0
Col: CentralAir 0
Col: Electrical 0
Col: 1stFlrSF 0
Col: 2ndFlrSF 0
Col: LowQualFinSF 0
Col: GrLivArea 0
Col: BsmtFullBath 4
Col: BsmtHalfBath 4
Col: FullBath 0
Col: HalfBath 0
Col: BedroomAbvGr 0
Col: KitchenAbvGr 0
Col: KitchenQual 2
Col: TotRmsAbvGrd 0
Col: Functional 4
Col: Fireplaces 0
Col: FireplaceQu 1460
Col: GarageType 152
Col: GarageYrBlt 156
Col: GarageFinish 156
Col: GarageCars 2
Col: GarageArea 2
Col: GarageQual 156
Col: GarageCond 156
Col: PavedDrive 0
Col: WoodDeckSF 0
Col: OpenPorchSF 0
Col: EnclosedPorch 0
Col: 3SsnPorch 0
Col: ScreenPorch 0
Col: PoolArea 0
Col: PoolQC 2912
Col: Fence 2338
Col: MiscFeature 2816
Col: MiscVal 0
Col: MoSold 0
Col: YrSold 0
Col: SaleType 2
Col: SaleCondition 0
"""

#fix NaN's

#remove for now FireplaceQu and MiscFeataure, too much NaN's...but theay are very important feature for price!?

ds = ds.drop(['FireplaceQu','MiscFeature'], axis=1)

#for OneHotEncoder
oCol=[]
nCol=[]

for col in ds:
    if ds[col].dtype=="O":
        oCol.append(col)
    else:
        nCol.append(col)
    ds[col]=ds[col].replace(np.nan,ds[col].mode()[0])
    
ds_enc=pd.get_dummies(ds, columns=oCol)


    



