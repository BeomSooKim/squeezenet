#%%
from aim.io.hdf5datasetwriter import HDF5DatasetWriter
from glob import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os, json
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import grey2rgb
import cv2
import numpy as np
import progressbar
#%%
data_path = 'D:/python/dataset/tiny-imagenet-200/train'
save_path = 'D:/python/dataset/tiny-imagenet-200/'
target_size = (64, 64)
img_paths = glob(os.path.sep.join([data_path, '*/*/*.jpeg']))
print('total images : {}'.format(len(img_paths)))
img_labels = [x.split(os.path.sep)[-3] for x in img_paths]
print('total labels : {}'.format(len(set(img_labels))))

encoder = LabelEncoder()
img_labels_en = encoder.fit_transform(img_labels)
trainX, valX, trainY, valY = train_test_split(img_paths, img_labels_en, stratify = img_labels_en,\
    shuffle = True, random_state = 35343, test_size = 0.2)

#%%
dataset = [('train', trainX, trainY),('val', valX, valY)]
for name, x, y in dataset:

    writer = HDF5DatasetWriter(dims = (len(x),*target_size, 3), \
    outputPath = os.path.sep.join([save_path, name + '_zero_centered.hdf5']))

    widgets = ["Building Dataset: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(x), widgets=widgets).start()

    for (i, (path, label)) in enumerate(zip(x, y)):
        img = imread(path)
        if len(img.shape) == 2:
            img = grey2rgb(img)
        img = np.array(img).astype(np.float32) - 127.5
        writer.add([img], [label])
        pbar.update(i)
    
    pbar.finish()
    writer.close()
#%%
class_dict = {}
for i, l in enumerate(encoder.classes_):
    class_dict[i] = l

with open('./class_dict.json', 'w') as f:
    f.write(json.dumps(class_dict))