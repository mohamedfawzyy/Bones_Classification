#!/usr/bin/env python
# coding: utf-8

# In[5]:


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


# In[2]:


train_path='data/train'
valid_path='data/valid'
train_metadata='MURA-v1.1/train_image_paths.csv'
valid_metadata='MURA-v1.1/valid_image_paths.csv'


# In[26]:


import os
print(os.path.basename(train_metadata))


# In[10]:


def load_data(path,destination,data_type):
    x=0
    images_paths=pd.read_csv(path)
    images_paths=pd.DataFrame(images_paths)
    for image_path in images_paths.loc[:,data_type]:
        img=Image.open(image_path)
        if 'negative' in image_path and 'XR_ELBOW' in image_path  :
          img.save(destination+"/negative/"+str(x)+ntpath.basename(image_path))
        elif 'positive' in image_path and 'XR_ELBOW' in image_path :
          img.save(destination+"/positive/"+str(x)+ntpath.basename(image_path))
        x=x+1


# In[11]:


load_data(train_metadata,'Data222/Elbow','train_image_paths')


# In[16]:


data_path='Data222/Elbow'
batch_size=1
target_size=(224,224)


# In[17]:


train_datagen = ImageDataGenerator(rescale=1. / 255,width_shift_range=0.2,height_shift_range=0.2,fill_mode='constant',cval=0.0,rotation_range=90)
datagenerator = train_datagen.flow_from_directory(data_path,
                                            target_size=target_size,
                                            color_mode='rgb' ,classes=["negative","positive"],
                                            batch_size = 1,
                                               
                                            class_mode='binary')
print("classes: ",datagenerator.class_indices)


# In[19]:


plt.figure(figsize=(10,10))
for i in range(0,3):
  for j in range(0,3):
    image = datagenerator[1][0][0]
    plt.subplot2grid((3,3),(i,j))
    plt.imshow(image)
# show the plot
plt.show()
print(datagenerator[1][1])


# In[20]:


from keras.preprocessing.image import array_to_img


# In[21]:


x=0
for i in range(0 , 4931):
    for j in range(0,2):
        x=x+1;
        image = datagenerator[i][0][0]
        image=array_to_img(image)
        label= datagenerator[i][1]
        if label ==1:
            image.save('Balance Data/Elbow'+'/positive/'+'img'+str(x)+'.png')
        else :   
            image.save('Balance Data/Elbow'+'/negative/'+'img'+str(x)+'.png')
            
            


# In[22]:


def count_samples(directory):
  numSamples=len(os.listdir(directory+"/positive"))+len(os.listdir(directory+"/negative"))
  return numSamples;


# In[23]:


num=count_samples('Balance Data/Elbow')
num


# In[6]:


numSamples1=len(os.listdir('Balance Data/Elbow'+"/positive"))
numSamples1


# In[139]:


numSamples11=len(os.listdir(destination1))
numSamples11


# In[94]:


numSamples2=len(os.listdir('Balance Data/Finger'+"/negative"))
numSamples2


# In[85]:


destination1='Balance Data2/positive'
destination2='Balance Data2/negative'


# In[84]:


from PIL import Image
import glob


# In[136]:


source='Balance Data/wrist/positive'
bone_type='wrist'


# In[137]:


def transferImages(source,bone_type,destination):
    for filename in glob.glob(source+'/*.png'):
        img=Image.open(filename)
        img.save(destination+'/'+bone_type+str(os.path.basename(filename)))


# In[138]:


transferImages(source,bone_type,destination1)


# In[140]:


import shutil

total, used, free = shutil.disk_usage("/")


print("Total: %d GiB" % (total // (2**30)))
print("Used: %d GiB" % (used // (2**30)))
print("Free: %d GiB" % (free // (2**30)))


# In[4]:


import numpy as np
import matplotlib.pyplot as plt

# data to plot
n_groups = 7
positive = (2006, 1968, 661,1484,599,4168,3987 )
negative = (2925, 3138, 1164, 4059,673,4211,5765)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, positive, bar_width,
alpha=opacity,
color='b',
label='positive')

rects2 = plt.bar(index + bar_width, negative, bar_width,
alpha=opacity,
color='g',
label='negative')

plt.xlabel('Bone')
plt.ylabel('Number of Images')
plt.title('With Out Data Augmentation')
plt.xticks(index + bar_width, ('Elbow', 'Finger', 'Forearm', 'Hand','Humerus','Shoulder','Wrist'))
plt.legend()

plt.tight_layout()
plt.show()


# In[22]:


numSamples1=len(os.listdir('Balance Data/wrist'+"/negative"))
numSamples1


# In[23]:


import numpy as np
import matplotlib.pyplot as plt

# data to plot
n_groups = 7
positive = (4012, 3937, 2644,4452,2996,4168,3987 )
negative = (5850, 3138, 4656, 4059,3366,4211,5765)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, positive, bar_width,
alpha=opacity,
color='b',
label='positive')

rects2 = plt.bar(index + bar_width, negative, bar_width,
alpha=opacity,
color='g',
label='negative')

plt.xlabel('Bone')
plt.ylabel('Number of Images')
plt.title('With Data Augmentation')
plt.xticks(index + bar_width, ('Elbow', 'Finger', 'Forearm', 'Hand','Humerus','Shoulder','Wrist'))
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:




