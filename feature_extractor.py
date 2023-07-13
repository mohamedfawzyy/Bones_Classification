#!/usr/bin/env python
# coding: utf-8

# In[2]:


import zipfile
#!pip install tensorflow==1.15
#!pip install --upgrade pip
#!pip install tensorflow-gpu==1.15  # GPU
#!pip install --user --upgrade tensorboard
#!pip install keras==2.3.1
import pandas as pd
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from numpy import save
from numpy import savetxt
from numpy import load
import os
import ntpath
get_ipython().system('pip install -U keras')
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization,Conv2D,MaxPooling2D
from keras import regularizers
from keras import models
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint
from keras.applications.mobilenet import MobileNet 
from keras.applications.mobilenet import preprocess_input 
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD as SVD
from sklearn.utils import class_weight
from sklearn import metrics
from tensorflow.python.keras.models import load_model
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
from tensorflow.keras import Input


# In[11]:


train_path='data/train'
balance_path='Balance Data2'
valid_path='data/valid'
train_metadata='MURA-v1.1/train_image_paths.csv'
valid_metadata='MURA-v1.1/valid_image_paths.csv'


# In[3]:


def count_samples(directory):
  numSamples=len(os.listdir(directory+"/positive"))+len(os.listdir(directory+"/negative"))
  return numSamples;


# In[4]:


mobileNet=MobileNet(input_shape=(224,224,3),weights='imagenet',include_top=False)
mobileNet.summary()


# In[5]:


outputs=mobileNet.layers[-1].output
outputs=layers.GlobalAveragePooling2D()(outputs)
base_model=models.Model(mobileNet.input,outputs)
base_model.summary()


# In[6]:


batch_size=128
target_size=(224,224)


# In[7]:


import shutil

total, used, free = shutil.disk_usage("/")

print("Total: %d GiB" % (total // (2**30)))
print("Used: %d GiB" % (used // (2**30)))
print("Free: %d GiB" % (free // (2**30)))


# In[8]:


train_datagen = ImageDataGenerator(rescale=1. / 255)#,zoom_range=0.1,width_shift_range=0.2,height_shift_range=0.2,fill_mode='constant',cval=0.0,rotation_range=20,horizontal_flip=True)
valid_datagen = ImageDataGenerator(rescale=1. / 255)


# In[9]:


def extract_train_features(directory):
    sample_count =count_samples(directory)  
    features = np.zeros(shape=(sample_count,1024))  # Must be equal to the output of the convolutional base
    labels = np.zeros(shape=(sample_count))
    # Preprocess data
    generator = train_datagen.flow_from_directory(directory,
                                            target_size=target_size,
                                            color_mode='rgb' ,classes=["negative","positive"],
                                            batch_size = batch_size,
                                            class_mode='binary')
    print("classes: ",generator.class_indices)

    '''
    # Pass data through convolutional base
    plt.figure(figsize=(8, 8))
    x,y = generator.next()
    k=0
    for i in range(0,4):
        for j in range(0,4):
            image = x[k]
            plt.subplot2grid((4,4),(i,j))
            plt.imshow(image)
            k = k+1
    # show the plot
    plt.show()'''
    i=0
    for inputs_batch, labels_batch in generator:
        features_batch = base_model.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
          print("i will break ")
          break
        print(i*batch_size)    
    return features, labels


# In[13]:


def extract_valid_features(directory):
    sample_count =count_samples(directory)  
    features = np.zeros(shape=(sample_count,1024))  # Must be equal to the output of the convolutional base
    labels = np.zeros(shape=(sample_count))
    # Preprocess data
    generator = valid_datagen.flow_from_directory(directory,
                                            target_size=target_size,
                                            color_mode='rgb' ,classes=["negative","positive"],
                                            batch_size = batch_size,
                                            class_mode='binary')
    
    print("classes: ",generator.class_indices)
    # Pass data through convolutional base
    
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = base_model.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
          print("i will break ")
          break
        print(i*batch_size)     
    return features, labels


# In[12]:


train_features, train_labels = extract_train_features(balance_path)
np.save('data3' + "/train_features",train_features)
np.save('data3' + "/train_labels",train_labels)


# In[14]:


valid_features, valid_labels = extract_valid_features(valid_path)
test_features,valid_features,test_labels,valid_labels=train_test_split(valid_features,valid_labels,train_size=0.1,random_state=1)


# In[15]:


np.save('data3' + "/valid_features",valid_features)
np.save('data3' + "/valid_labels",valid_labels)
np.save('data3' + "/test_features",test_features)
np.save('data3' + "/test_labels",test_labels)


# In[16]:


def load_features(path):
  train_features=np.load(path+"/train_features.npy" )
  train_labels=np.load(path+"/train_labels.npy" )
  test_features=np.load(path+"/test_features.npy" )
  test_labels=np.load(path+"/test_labels.npy" )
  valid_features=np.load(path+"/valid_features.npy" )
  valid_labels=np.load(path+"/valid_labels.npy" )
  return train_features,train_labels,test_features,test_labels,valid_features,valid_labels;


# In[17]:


train_features,train_labels,test_features,test_labels,valid_features,valid_labels=load_features('data3')
print("shape of train  features ",train_features.shape)
print("shape of train   labels ",train_labels.shape)
print("shape of test  features ",test_features.shape)
print("shape of test   labels ",test_labels.shape)
print("shape of validation  features ",valid_features.shape)
print("shape of validation   labels ",valid_labels.shape)


# In[85]:


from keras import regularizers
input_shape =1024
model = models.Sequential()
model.add(layers.InputLayer(input_shape=(input_shape,)))
model.add(layers.Dense(512, activation='relu', input_dim=input_shape))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


# In[86]:


model.summary()


# In[87]:


optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
lr_list=list()
#optimizer=optimizers.SGD(learning_rate=0.001)
#optimizer=optimizers.Adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-07, name="Adadelta")


# In[88]:


def scheduler(epoch, lr):
  lr_list.append(lr)
  if epoch % 3 == 0 and epoch != 0:
    return lr * 0.1
  else:   
    return lr 


# In[89]:


from keras.callbacks import LearningRateScheduler
reduce_lr=LearningRateScheduler(scheduler)
earlyStopping=EarlyStopping(monitor='val_loss',patience=5,mode='min')
#reduce_lr=ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=2,verbose=1,mode='min')
filepath="data3/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')


# In[90]:


model.compile( optimizer=optimizer,loss='binary_crossentropy',metrics=['acc'])


# In[91]:


history=model.fit(x=train_features,y=train_labels, batch_size=128,epochs=30,verbose=1,callbacks=[checkpoint,reduce_lr,earlyStopping],validation_data=(valid_features,valid_labels),validation_freq=1,shuffle=True)


# In[92]:


model.load_weights("data3/weights.best.hdf5")


# In[93]:


model.evaluate(test_features,test_labels,verbose=1,batch_size=128)


# In[94]:


model.save("data2/feature_extractor.h5")


# In[95]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'g', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.figure()

plt.plot(epochs, lr_list, 'g', label='learning_rate')
plt.title('learning rate decay')
plt.legend()

plt.show()


# In[96]:


from sklearn.metrics import f1_score
y_pred=model.predict_classes(test_features)
f1_Score=f1_score(test_labels, y_pred, average='binary')


# In[97]:


print("F1_score: " ,f1_Score)


# In[ ]:




