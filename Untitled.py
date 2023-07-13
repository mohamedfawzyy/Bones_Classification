#!/usr/bin/env python
# coding: utf-8

# In[1]:


import zipfile
#!pip install tensorflow==1.13.1
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
#!pip install keras==2.3.1
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization,Conv2D,MaxPooling2D
from keras import regularizers
from keras import models
from keras import optimizers
from keras.callbacks import LearningRateScheduler,EarlyStopping,ModelCheckpoint
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


# In[3]:


get_ipython().system('unzip -q "MURA-v1.1.zip"')


# In[3]:


os.makedirs('data')
os.makedirs('data/train')
os.makedirs('data/valid')


# In[5]:


os.makedirs('data/train/positive')
os.makedirs('data/train/negative')
os.makedirs('data/valid/positive')
os.makedirs('data/valid/negative')


# In[2]:


train_path='data/train'
valid_path='data/valid'
train_metadata='MURA-v1.1/train_image_paths.csv'
valid_metadata='MURA-v1.1/valid_image_paths.csv'


# In[3]:


def load_data(path,destination,data_type):
    x=0
    images_paths=pd.read_csv(path)
    images_paths=pd.DataFrame(images_paths)
    for image_path in images_paths.loc[:,data_type]:
        img=Image.open(image_path)
        if image_path.find("positive")==-1:
          img.save(destination+"/negative/"+str(x)+ntpath.basename(image_path))
        else:
          img.save(destination+"/positive/"+str(x)+ntpath.basename(image_path))
        x=x+1


# In[14]:


load_data(train_metadata,train_path,'train_image_paths')


# In[47]:


load_data(valid_metadata,valid_path,'valid_image_paths')


# In[3]:


def count_samples(directory):
  numSamples=len(os.listdir(directory+"/positive"))+len(os.listdir(directory+"/negative"))
  return numSamples;


# In[3]:


import shutil

total, used, free = shutil.disk_usage("/")


print("Total: %d GiB" % (total // (2**30)))
print("Used: %d GiB" % (used // (2**30)))
print("Free: %d GiB" % (free // (2**30)))


# In[4]:


#run the model from scratch


# In[4]:


batch_size=128
target_size=(224,224)


# In[5]:


train_datagen = ImageDataGenerator(rescale=1. / 255,zoom_range=0.1,width_shift_range=0.2,height_shift_range=0.2,fill_mode='constant',cval=0.0,rotation_range=90,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)
datagenerator = train_datagen.flow_from_directory(train_path,
                                            target_size=target_size,
                                            color_mode='rgb' ,classes=["negative","positive"],
                                            batch_size = 128,
                                            class_mode='binary')
print("classes: ",datagenerator.class_indices)


# In[6]:


def get_test_data(directory):
    # Preprocess data
    sample_count =count_samples(directory)  
    testdata = np.zeros(shape=(sample_count, 224,224, 3))  # Must be equal to the output of the convolutional base
    labels = np.zeros(shape=(sample_count))
    testgenerator = test_datagen.flow_from_directory(directory,
                                            target_size=target_size,
                                            color_mode='rgb' ,classes=["negative","positive"],
                                            batch_size = batch_size,
                                            class_mode='binary')
    print("classes: ",testgenerator.class_indices)
    i = 0
    for inputs_batch, labels_batch in testgenerator:
        #features_batch = model.predict(inputs_batch)
        testdata[i * batch_size: (i + 1) * batch_size] = inputs_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
          print("i will break ")
          break
        print(i*batch_size) 
    return testdata,labels;


# In[11]:


testdata,testlabels=get_test_data(valid_path)
testdata,validdata,testlabels,validlabels=train_test_split(testdata,testlabels,train_size=0.2,random_state=1)


# In[12]:


np.save("testdata",testdata)
np.save("testlabels",testlabels)
np.save("validdata",validdata)
np.save("validlabels",validlabels)
print("shape of testdata ",testdata.shape)
print("shape of testlabels",testlabels.shape)
print("shape of validdata ",validdata.shape)
print("shape of validlabels ",validlabels.shape)


# In[7]:


validdata=np.load('validdata.npy')
validlabels=np.load('validlabels.npy')
print("shape of validdata ",validdata.shape)
print("shape of validlabels ",validlabels.shape)


# In[8]:


#create model 
model = models.Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(224, 224, 3)))
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(1024, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid',))
model.summary()


# In[9]:


lr_list=list()
def scheduler(epoch, lr):
   
    lr_list.append(lr)  
    if epoch % 12 == 0 and epoch != 0:
       
        return lr * 0.1
    else:
        
        return lr 


# In[10]:


optimizer=optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
reduce_lr=LearningRateScheduler(scheduler)
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.compile( optimizer=optimizer,loss='binary_crossentropy',metrics=['acc'])


# In[11]:


model.load_weights("weights.best.hdf5")
History=model.fit(datagenerator,epochs=60,verbose=1,steps_per_epoch=len(datagenerator), initial_epoch=54,callbacks=[reduce_lr,checkpoint],validation_data=(validdata  ,validlabels ),validation_freq=1,shuffle=True)


# In[21]:


model.save('scratchModel (2).h5')


# In[ ]:


test_data=np.load('testdata.npy')
test_labels=np.load('testlabels.npy')
model.evaluate(test_data,test_labels,verbose=1,batch_size=128)


# In[23]:


acc = History.history['acc']
val_acc = History.history['val_acc']
loss = History.history['loss']
val_loss = History.history['val_loss']

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

#plt.figure()

#plt.plot(epochs, lr_list, 'g', label='learning_rate')
#plt.title('learning rate decay')
#plt.legend()


# In[ ]:




