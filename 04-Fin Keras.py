# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 11:07:36 2022

@author: mladjan.jovanovic
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras import metrics
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from keras.models import load_model

import joblib

#fix reproduciblity?
seed_value = 0
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
keras.utils.set_random_seed(seed_value)    
    
train_ds = pd.read_csv('./train.csv', sep=',')
test_ds = pd.read_csv('./test.csv', sep=',')

# train_ds.head()
# train_ds.info()

ds=pd.concat([train_ds,test_ds], ignore_index=True)

# for col in ds:
#     if ds[col].isna().sum() > 0: print('Col:',col,str(ds[col].isna().sum()))

hId=ds.loc[1460:2919,'Id'] #for finall output
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

y = ds_enc.loc[0:1460, 'SalePrice']
X = ds_enc.loc[0:1460, :]
X = X.drop('SalePrice', axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1337)



def basic_model_3(x_size, y_size):
    t_model = Sequential()
    t_model.add(Dense(80, activation="tanh", kernel_initializer='normal', input_shape=(x_size,)))
    t_model.add(Dropout(0.2))
    t_model.add(Dense(120, activation="relu", kernel_initializer='normal', 
        kernel_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1(0.01)))
    t_model.add(Dropout(0.1))
    t_model.add(Dense(20, activation="relu", kernel_initializer='normal', 
        kernel_regularizer=regularizers.l1_l2(0.01), bias_regularizer=regularizers.l1_l2(0.01)))
    t_model.add(Dropout(0.1))
    t_model.add(Dense(10, activation="relu", kernel_initializer='normal'))
    t_model.add(Dropout(0.0))
    t_model.add(Dense(y_size))
    t_model.compile(
        loss='mean_squared_error',
        optimizer='nadam',
        metrics=[metrics.mae])
    return(t_model)

def basic_model_4(x_size, y_size):
    t_model = Sequential()
    t_model.add(Dense(279, activation="relu", input_shape=(x_size,)))
    t_model.add(Dropout(0.1))
    t_model.add(Dense(100, activation="relu"))
    t_model.add(Dense(50, activation="relu"))
    t_model.add(Dense(y_size))
    t_model.compile(loss='mean_squared_error',
        optimizer=Adam(),
        metrics=[metrics.mae])
    return(t_model)

def basic_model_5(x_size, y_size):
    t_model = Sequential()
    t_model.add(Dense(279, activation="elu", input_shape=(x_size,)))
    t_model.add(Dropout(0.1))
    t_model.add(Dense(100, activation="elu"))
    t_model.add(Dense(50, activation="elu"))
    t_model.add(Dense(y_size))
    t_model.compile(loss='mean_squared_error',
        optimizer=Adam(),
        metrics=[metrics.mae])
    return t_model


def basic_model_6(x_size, y_size):
    t_model = Sequential()
    t_model.add(Dense(300, activation="elu", input_shape=(x_size,)))
    t_model.add(Dropout(0.1))
    t_model.add(Dense(150, activation="elu"))
    t_model.add(Dense(50, activation="elu"))
    t_model.add(Dense(25, activation="elu"))
    t_model.add(Dense(y_size))
    t_model.compile(loss='mean_squared_error',
        optimizer=Adam(),
        metrics=[metrics.mae])
    return t_model

def basic_model_7(x_size, y_size):
    t_model = Sequential()
    t_model.add(Dense(300, activation="elu", input_shape=(x_size,)))
    t_model.add(Dropout(0.1))
    t_model.add(Dense(200, activation="elu"))
    t_model.add(Dropout(0.1))
    t_model.add(Dense(50, activation="elu"))
    t_model.add(Dense(25, activation="elu"))
    t_model.add(Dense(y_size))
    t_model.compile(loss='mean_squared_error',
        optimizer=Adam(),
        metrics=[metrics.mae])
    return t_model

def basic_model_8(x_size, y_size):
    t_model = Sequential()
    t_model.add(Dense(400, activation="elu", input_shape=(x_size,)))
    t_model.add(Dropout(0.1))
    t_model.add(Dense(200, activation="elu"))
    t_model.add(Dropout(0.1))
    t_model.add(Dense(10, activation="elu"))
    t_model.add(Dense(50, activation="elu"))
    t_model.add(Dense(y_size))
    t_model.compile(loss='mean_squared_error',
        optimizer=Adam(),
        metrics=[metrics.mae])
    return t_model

model = basic_model_8(X_train.shape[1], y_train.shape[0])
#model.summary()

epochs = 500
batch_size = 128

print('Epochs: ', epochs)
print('Batch size: ', batch_size)

keras_callbacks = [
    # ModelCheckpoint('/tmp/keras_checkpoints/model.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', save_best_only=True, verbose=2)
    # ModelCheckpoint('/tmp/keras_checkpoints/model.{epoch:02d}.hdf5', monitor='val_loss', save_best_only=True, verbose=0)
    # TensorBoard(log_dir='/tmp/keras_logs/model_3', histogram_freq=0, write_graph=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None),
    EarlyStopping(monitor='val_mean_absolute_error', patience=20, verbose=1) #1 to print when is triggered ES
]


history = model.fit(X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
    verbose=0, #2 for detailed output
    validation_data=(X_test, y_test),
    callbacks=keras_callbacks)


train_score = model.evaluate(X_train, y_train, verbose=0) #0 to suppress print
test_score = model.evaluate(X_test, y_test, verbose=0) #0 to suppress print

"""
print(model.metrics_names)
['loss', 'mean_absolute_error']
"""""

print('Train MAE: ', round(train_score[1], 0), ', Train Loss: ', round(train_score[0], 0)) 
print('Val MAE: ', round(test_score[1], 0), ', Val Loss: ', round(test_score[0], 0))

"""
print(history.history.keys())
dict_keys(['loss', 'mean_absolute_error', 'val_loss', 'val_mean_absolute_error'])
"""

def plot_hist(h, xsize=6, ysize=10):
    # Prepare plotting
    fig_size = plt.rcParams["figure.figsize"]
    plt.rcParams["figure.figsize"] = [xsize, ysize]
    fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True)
    
    # summarize history for MAE
    plt.subplot(211)
    plt.plot(h['mean_absolute_error'])
    plt.plot(h['val_mean_absolute_error'])
    plt.title('Training vs Validation MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # summarize history for loss
    plt.subplot(212)
    plt.plot(h['loss'])
    plt.plot(h['val_loss'])
    plt.title('Training vs Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot it all in IPython (non-interactive)
    plt.draw()
    plt.show()
    return

plot_hist(history.history, xsize=8, ysize=12)

y_res = ds_enc.loc[1460:2919, 'SalePrice']
X_res = ds_enc.loc[1460:2919, :]
X_res = X_res.drop('SalePrice', axis=1)

y_res=model.predict(X_res)

res=pd.DataFrame(hId)
res['SalePrice'] = y_res[:,0]

save_res=res.to_csv('./sample_submission.csv', index=False)