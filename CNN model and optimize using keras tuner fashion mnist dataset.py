#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install keras-tuner')


# In[2]:


import tensorflow as tf
from tensorflow import keras
import numpy as np


# In[3]:


tf.__version__


# In[4]:


fashion_mnist=keras.datasets.fashion_mnist


# In[5]:


(train_images, train_labels),(test_images,test_labels)=fashion_mnist.load_data() #data preprocessing


# In[6]:


train_images=train_images/255.0
test_images=test_images/255.0


# In[7]:


train_images=train_images.reshape(len(train_images),28,28,1)
test_images=test_images.reshape(len(test_images),28,28,1)


# In[11]:


def build_model(hp):  
  model = keras.Sequential([
    keras.layers.Conv2D(
        filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
        kernel_size=hp.Choice('conv_1_kernel', values = [3,5]),
        activation='relu',
        input_shape=(28,28,1)
    ),
    keras.layers.Conv2D(
        filters=hp.Int('conv_2_filter', min_value=32, max_value=64, step=16),
        kernel_size=hp.Choice('conv_2_kernel', values = [3,5]),
        activation='relu'
    ),
    keras.layers.Flatten(),
    keras.layers.Dense(
        units=hp.Int('dense_1_units', min_value=32, max_value=128, step=16),
        activation='relu'
    ),
    keras.layers.Dense(10, activation='softmax')
  ])
  
  model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
  
  return model


# In[12]:


from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters


# In[13]:


tuner_search=RandomSearch(build_model,
                          objective='val_accuracy', 
                          max_trials=5, 
                          directory='output',
                          project_name='mnist fashion')


# In[ ]:


tuner_search.search(train_images,train_labels,epochs=3,validation_split=0.1)


# In[ ]:


model=tuner_search.get_best_models(num_models=1)[0]


# In[ ]:


model.summary()


# In[ ]:


model.fit(train_images, train_labels, epochs=10, validation_split=0.1, initial_epoch=3)


# In[ ]:





# In[ ]:




