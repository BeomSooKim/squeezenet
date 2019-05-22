#%%
from keras.preprocessing.image import ImageDataGenerator
from aim.io import HDF5DatasetGenerator
import matplotlib.pyplot as plt 
aug = ImageDataGenerator(rescale = 1.)
data_path = 'D:\\python\\dataset\\tiny-imagenet-200\\val.hdf5'
noaug = HDF5DatasetGenerator(data_path, batchSize = 4, classes = 200)
withaug = HDF5DatasetGenerator(data_path, batchSize = 4, classes = 200, aug = aug)
#%%
#no aug
imgs, labels = next(noaug.generator())
imgs = imgs.astype(np.uint8)
for i in range(4):
    plt.imshow(imgs[i])
    plt.show()
#%%
#with aug
imgs, labels = next(withaug.generator())
imgs = imgs.astype(np.uint8)
for i in range(4):
    plt.imshow(imgs[i])
    plt.show()