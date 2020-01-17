from keras.models import Sequential, Model
from keras.layers import Dense, Input, Reshape, Conv2DTranspose,UpSampling2D,Flatten
import numpy as np

def build_discriminator():
    model = Sequential()
    model.add(Dense(32*16*3, input_shape=(64*16*3,),activation='relu'))
    model.add(Dense(16*16*3,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    input_ = Input(shape=(64*16*3,))
    output_ = model(input_)
    return Model(input_, output_)


def build_generator():
    model = Sequential()
    model.add(Reshape((3, 4, 512), input_shape=(3 * 4 * 512,)))
    model.add(Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu'))
    model.add(Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu'))
    model.add(Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu'))
    model.add(Conv2DTranspose(filters=1, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='tanh'))
    model.add(Flatten())
    model.summary()
    input_ = Input(shape=(3*4*512,))
    output_ = model(input_)
    return Model(input_, output_)

if __name__ == '__main__':
    #Attack_free_dataset = np.load("Attack_free_dataset.npy")
    #DoS_attack_dataset = np.load("DoS_attack_dataset.npy")
    #Fuzzy_attack_dataset = np.load("Fuzzy_attack_dataset.npy")
    #Impersonation_attack_dataset = np.load("Impersonation_attack_dataset.npy")

    Attack_free_dataset = np.random.normal(0, 1, (128*5, 3*4*512))
    DoS_attack_dataset = np.random.normal(0, 1, (128*5, 3*4*512))
    Fuzzy_attack_dataset = np.random.normal(0, 1, (128*5, 3*4*512))
    Impersonation_attack_dataset = np.random.normal(0, 1, (128*5, 3*4*512))

    discriminator = build_discriminator()
    discriminator.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    generator = build_generator()
    z = Input(shape=(3*4*512,))
    img = generator(z)
    discriminator.trainable = False
    validity = discriminator(img)
    combined = Model(z, validity)
    combined.compile(loss='binary_crossentropy', optimizer='Adam')

    batch_size = 128
    epochs = 1000
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        idx = np.random.randint(0, Attack_free_dataset.shape[0], batch_size)
        imgs = Attack_free_dataset[idx]
        noise = np.random.normal(0, 1, (batch_size, 3*4*512))
        gen_imgs = generator.predict(noise)
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, 3*4*512))
        g_loss = combined.train_on_batch(noise, valid)
        print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

    print(discriminator.evaluate(DoS_attack_dataset,np.zeros((len(DoS_attack_dataset),1)),batch_size=64))
    print(discriminator.evaluate(Fuzzy_attack_dataset, np.zeros((len(Fuzzy_attack_dataset), 1)),batch_size=64))
    print(discriminator.evaluate(Impersonation_attack_dataset, np.zeros((len(Impersonation_attack_dataset), 1)),batch_size=64))
