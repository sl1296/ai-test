from keras.models import Sequential
from keras.layers.convolutional import Conv2DTranspose
from keras.layers import Reshape,UpSampling2D
def build_generator():
    model = Sequential()
    model.add(Reshape((3, 4, 512), input_shape=(3*4*512,)))
    model.add(Conv2DTranspose(filters=256, kernel_size=(4, 4),strides=(2, 2), padding='same', activation='relu'))
    model.add(Conv2DTranspose(filters=128,kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu'))
    model.add(Conv2DTranspose(filters=64,kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu'))
    model.add(Conv2DTranspose(filters=1,kernel_size=(4, 4), strides=(2, 2), padding='same', activation='tanh'))
    print(model.summary())
    input_ = Input(shape=(3*4*512,))
    output_ = model(input_)
    return Model(input_, output_)
build_generator()