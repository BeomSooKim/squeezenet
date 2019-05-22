#%% 
from keras.models import load_model

import h5py
import numpy as np 
from ranked.topn_accuracy import rank_n_accuracy

from sklearn.metrics import accuracy_score
#%%
model = load_model('D:\\AI_Research\squeezenet\\model\\xavier\\m_138-0.3099-3.1618.hdf5')

val = h5py.File("D:\python\dataset\\tiny-imagenet-200\\val.hdf5")
labels = np.array(val['labels'])
images = np.array(val['images'])

pred = model.predict(images / 255.0, batch_size = 512, verbose = 1)
rank_n_accuracy(labels, pred, 5)

#%%
model = load_model("D:\\AI_Research\squeezenet\\model\\xavier\\m_138-0.3099-3.1618.hdf5")
val = h5py.File("D:\python\dataset\\tiny-imagenet-200\\val.hdf5")
labels = np.array(val['labels'])
images = np.array(val['images'])
images = (images - 127.5) / 127.5

pred = model.predict(images, batch_size = 512, verbose = 1)
rank_n_accuracy(labels, pred, 5)

