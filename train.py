#%%
import os, json
import matplotlib.pyplot as plt
import numpy as np
import timeit

from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.activations import relu
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
import keras.backend as K

from aim.nn.conv.squeezenet import SqueezeNet
#from aim.nn.conv.sqn import SqueezeNet
from aim.nn.conv.lenet import LeNet
from aim.nn.conv.vgglike2 import VGGLike2
from aim.io.hdf5datasetgenerator import HDF5DatasetGenerator
import matplotlib as mp
mp.rcParams['figure.facecolor'] = 'white'
#%%
#K.set_floatx('float64')
base_lr = 0.04
initial_epoch = 0
n_epochs = 5
batch_size = 512
with open('D:\AI_Research\squeezenet\class_dict.json', 'r') as f:
#with open("D:\\python\\blueward_custom10000\\data\\class_dict.json", 'r') as f:
    class_dict = json.load(f)
num_class = len(class_dict.keys())
save_dir = 'D:\\AI_Research\\squeezenet\\model'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

def linear_decay(epoch):
    return (1- epoch / n_epochs)*base_lr
#    return -(base_lr / n_epochs) * epoch + base_lr
#%%
train_augmenter = ImageDataGenerator(rotation_range=15, zoom_range=0.1, width_shift_range= 0.1,\
    height_shift_range= 0.1, horizontal_flip= True, vertical_flip= True, rescale = 1/127.5)
val_augmenter = ImageDataGenerator(rescale = 1/127.5)

trainGen = HDF5DatasetGenerator('D:\python\dataset\\tiny-imagenet-200\\train_zero_centered.hdf5',\
    batchSize = batch_size, aug = train_augmenter, classes = num_class)
valGen = HDF5DatasetGenerator("D:\python\dataset\\tiny-imagenet-200\\val_zero_centered.hdf5",\
    batchSize = batch_size, aug = val_augmenter, classes = num_class)

#trainGen = HDF5DatasetGenerator('D:\python\\blueward_custom10000\\data\\train.hdf5',\
#    batchSize = batch_size, aug = val_augmenter, classes = num_class)
#valGen = HDF5DatasetGenerator("D:\python\\blueward_custom10000\\data\\val.hdf5",\
#    batchSize = batch_size, aug = val_augmenter, classes = num_class)

ckpt = ModelCheckpoint(filepath = os.path.sep.join([save_dir, 'm_{epoch:03d}-{val_acc:.4f}-{val_loss:.4f}.hdf5']),\
     monitor="val_loss", save_best_only= True, verbose = 1, period = 1)
lr_decay = LearningRateScheduler(linear_decay, verbose = 1)
reduce_lr = ReduceLROnPlateau(factor = 0.3, patience = 20)

#model = SqueezeNet(200, inputs = (64, 64, 3))
model = SqueezeNet.build((64, 64, 3), 96, 16, 64, 64, num_class, 0.00001)
#model = LeNet.build(64, 64, 3, 200)
#model = VGGLike2.build(64, 64, 3, 200)
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = SGD(base_lr, momentum = 0.9), metrics = ['accuracy'])


#%%
start = timeit.default_timer()
H = model.fit_generator(trainGen.generator(),\
    steps_per_epoch = trainGen.numImages // batch_size,\
    validation_data = valGen.generator(),\
    validation_steps = valGen.numImages // batch_size,\
    epochs = n_epochs,
    callbacks = [ckpt, reduce_lr])
end = timeit.default_timer()
print(end - start)
#%%
log = H.history
epochs = np.arange(n_epochs)
plt.figure(figsize = (16,8))
plt.style.use('seaborn')
plt.plot(epochs,log['loss'], label = 'train-loss')
plt.plot(epochs, log['val_loss'], label = 'val-loss')
plt.plot(epochs, log['acc'],label = 'train-acc')
plt.plot(epochs, log['val_acc'], label = 'val-acc')
plt.legend()
plt.savefig(os.path.sep.join([save_dir, 'loss_curve.jpg']), dpi = 300)
plt.show()

with open(os.path.sep.join([save_dir, 'loss_history.json']), 'w') as f:
    f.write(json.dumps(log))

#%%
