#%%
from keras.layers import Conv2D, MaxPooling2D, Concatenate, Softmax, Input, GlobalAveragePooling2D
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Dropout
from keras.layers import Activation

class SqueezeNet:
    # fire module
    @staticmethod
    def fire_module(x, s11, e11, e33, regularizer = 0.0005):
        squeeze = Conv2D(filters = s11, kernel_size = (1,1), strides = (1, 1), \
            kernel_regularizer = l2(regularizer), padding = 'same', kernel_initializer = 'glorot_normal')(x)
        squeeze = Activation('relu')(squeeze)
        expand_1x1 = Conv2D(filters = e11, kernel_size = (1,1), strides=(1,1), \
            kernel_regularizer=l2(regularizer), padding = 'same', kernel_initializer = 'glorot_normal')(squeeze)
        expand_1x1 = Activation('relu')(expand_1x1)
        expand_3x3 = Conv2D(filters = e11, kernel_size = (3,3), strides=(1,1), \
            kernel_regularizer=l2(regularizer), padding = 'same', kernel_initializer = 'glorot_normal')(squeeze)
        expand_3x3 = Activation('relu')(expand_3x3)
        out = Concatenate()([expand_1x1, expand_3x3])
        return out
    @staticmethod
    def build(input_shape, initial_filter, s11, e11, e33, n_class, regularizer):
        _input = Input(input_shape)

        #initial convolution layer
        x = Conv2D(filters = initial_filter, kernel_size = (7,7), strides = (2, 2),\
            kernel_regularizer= l2(regularizer), padding = 'same',\
            kernel_initializer = 'glorot_normal')(_input)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2))(x)
        # fire2
        x = SqueezeNet.fire_module(x, s11, e11, e33, regularizer)
        # fire3
        x = SqueezeNet.fire_module(x, s11, e11, e33, regularizer)
        #fire4
        x = SqueezeNet.fire_module(x, s11*2, e11*2, e33*2, regularizer)
        x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2))(x)

        #fire5
        x = SqueezeNet.fire_module(x, s11*2, e11*2, e33*2, regularizer)
        #fire6
        x = SqueezeNet.fire_module(x, s11*3, e11*3, e33*3, regularizer)
        #fire7
        x = SqueezeNet.fire_module(x, s11*3, e11*3, e33*3, regularizer)
        #fire8
        x = SqueezeNet.fire_module(x, s11*4, e11*4, e33*4, regularizer)
        x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2))(x)
        #fire9
        x = SqueezeNet.fire_module(x, s11*4, e11*4, e33*4, regularizer)
        x = Dropout(0.5)(x)
        x = Conv2D(filters = n_class, kernel_size = (1, 1), strides = (1, 1), \
            kernel_regularizer=l2(regularizer), padding = 'valid',\
            kernel_initializer = 'glorot_normal')(x)
        x = Activation('relu')(x)
        x = GlobalAveragePooling2D()(x)
        out = Softmax()(x)

        model = Model(_input, out)

        return model