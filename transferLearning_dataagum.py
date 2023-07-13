#!/usr/bin/env python
# coding: utf-8

# In[1]:


import zipfile
#!pip install tensorflow==2.2.0
#!pip install --upgrade pip
#!pip install tensorflow-gpu==2.2.0  # GPU
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
#!pip install -U keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization,Conv2D,MaxPooling2D
from keras import regularizers
from keras import models
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint,LearningRateScheduler
from keras.applications.mobilenet import MobileNet 
from keras.applications.mobilenet import preprocess_input 
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD as SVD
from sklearn.utils import class_weight
from sklearn import metrics
from keras.models import load_model
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
from tensorflow.keras import Input


# In[2]:


train_path='data/train'
valid_path='data/valid'
train_metadata='MURA-v1.1/train_image_paths.csv'
valid_metadata='MURA-v1.1/valid_image_paths.csv'


# In[3]:


def count_samples(directory):
  numSamples=len(os.listdir(directory+"/positive"))+len(os.listdir(directory+"/negative"))
  return numSamples;


# In[4]:


mobileNet=MobileNet(input_shape=(224,224,3),weights='imagenet',include_top=False)
outputs=mobileNet.layers[-1].output
outputs=layers.GlobalAveragePooling2D()(outputs)
base_model=models.Model(mobileNet.input,outputs)
base_model.summary()


# In[5]:


batch_size=128
target_size=(224,224)


# In[6]:


import shutil

total, used, free = shutil.disk_usage("/")

print("Total: %d GiB" % (total // (2**30)))
print("Used: %d GiB" % (used // (2**30)))
print("Free: %d GiB" % (free // (2**30)))


# In[7]:


train_datagen = ImageDataGenerator(rescale=1. / 255,zoom_range=0.1,width_shift_range=0.2,height_shift_range=0.2,fill_mode='constant',cval=0.0,rotation_range=90)
valid_datagen = ImageDataGenerator(rescale=1. / 255)


# In[8]:


balance_path='Balance Data2'


# In[9]:


train_generator = train_datagen.flow_from_directory(balance_path,
                                            target_size=target_size,        
                                            color_mode='rgb' ,classes=["negative","positive"],
                                            batch_size = batch_size,
                                            class_mode='binary')
print("classes: ",train_generator.class_indices)


# In[12]:


valid_generator = valid_datagen.flow_from_directory(valid_path,
                                            target_size=target_size,
                                            color_mode='rgb' ,classes=["negative","positive"],
                                            batch_size = batch_size,
                                            class_mode='binary')
print("classes: ",valid_generator.class_indices)
    


# In[10]:


def extract_valid_data(directory):
    sample_count =count_samples(directory)  
    data = np.zeros(shape=(sample_count,224,224,3))  # Must be equal to the output of the convolutional base
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
        data[i * batch_size: (i + 1) * batch_size] = inputs_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
          print("i will break ")
          break
        print(i*batch_size)     
    return data, labels


# In[11]:


valid_data, valid_labels = extract_valid_data(valid_path)


# In[12]:


testdata,validdata,testlabels,validlabels=train_test_split(valid_data,valid_labels,train_size=0.1,random_state=42)


# In[13]:


np.save("testdata",testdata)
np.save("testlabels",testlabels)
np.save("validdata",validdata)
np.save("validlabels",validlabels)
print("shape of testdata ",testdata.shape)
print("shape of testlabels",testlabels.shape)
print("shape of validdata ",validdata.shape)
print("shape of validlabels ",validlabels.shape)


# In[30]:


testdata=np.load("testdata.npy")
testlabels=np.load("testlabels.npy")
print("shape of testdata ",testdata.shape)
print("shape of testlabels ",testlabels.shape)


# In[10]:


validdata=np.load("validdata.npy")
validlabels=np.load("validlabels.npy")
print("shape of validdata ",validdata.shape)
print("shape of validlabels ",validlabels.shape)


# In[11]:


plt.imshow(validdata[510])
print(validlabels[510])


# In[12]:


for layer in base_model.layers:
    layer.trainable=False
for i, layer in enumerate(base_model.layers):
    layer._name='layer_'+str(i)
base_model.summary()        


# In[13]:


for i in range(78,87):
    base_model.get_layer('layer_'+str(i)).trainable =True
base_model.summary()


# In[14]:


input_shape = base_model.output_shape[1]
model = models.Sequential()
model.add(base_model)
model.add(layers.Dense(512, activation='relu',input_dim=input_shape))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.summary()


# In[38]:


model.save('TL_agumentation/model1.h5')


# In[15]:


optimizer=optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, amsgrad=False)
lr_list=list()


# In[16]:


def scheduler(epoch, lr):
  
    lr_list.append(lr)  
    if epoch % 8 == 0 and epoch != 0:
       
        return lr * 0.1
    else:
       
        return lr 


# In[17]:


#earlyStopping=EarlyStopping(monitor='val_loss',patience=8,mode='min')
reduce_lr=LearningRateScheduler(scheduler)
filepath="TL_agumentation/weights_TL_agum.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss',save_weights_only=True,verbose=1, save_best_only=True, mode='min')


# In[18]:


model.compile( optimizer=optimizer,loss='binary_crossentropy',metrics=['acc'])


# In[19]:


model.load_weights('TL_agumentation/weights_TL_agum.hdf5')
history=model.fit(train_generator,epochs=20,verbose=1,steps_per_epoch=len(train_generator),callbacks=[reduce_lr,checkpoint],initial_epoch=17,validation_data=(validdata,validlabels),validation_freq=1)


# In[27]:


model.save('TL_agumentation/model.h5')


# In[28]:


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


# In[31]:


model.evaluate(testdata,testlabels,verbose=1)


# In[32]:


model.evaluate(validdata,validlabels,verbose=1)


# In[ ]:




