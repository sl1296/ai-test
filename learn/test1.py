import numpy as np
from keras.models import Sequential,model_from_json
from keras.layers.core import Dense,Dropout,Activation
from keras.layers import MaxPooling2D,Flatten
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from keras.datasets import mnist
def load_data():
	(x_test,y_test),(x_train,y_train)=mnist.load_data()
	#number=10000
	#x_train=x_train[0:number]
	#y_train=y_train[0:number]
	print(x_test.shape)
	x_train=x_train.reshape(x_train.shape[0],28*28)
	x_test=x_test.reshape(x_test.shape[0],28*28)
	x_train=x_train.astype('float32')
	x_test=x_test.astype('float32')
	y_train=np_utils.to_categorical(y_train,10)
	y_test=np_utils.to_categorical(y_test,10)
	x_train=x_train/255
	x_test=x_test/255
	return (x_train,y_train),(x_test,y_test)
(x_train,y_train),(x_test,y_test)=load_data()

model=Sequential()
model.add(Dense(input_dim=28*28,units=700,activation='relu'))
model.add(Dense(units=700,activation='relu'))
model.add(Dense(units=700,activation='relu'))
model.add(Dense(units=700,activation='relu'))
model.add(Dense(units=700,activation='relu'))
model.add(Dense(units=700,activation='relu'))
model.add(Dense(units=700,activation='relu'))
model.add(Dense(units=700,activation='relu'))
model.add(Dense(units=700,activation='relu'))
model.add(Dense(units=700,activation='relu'))
model.add(Dense(units=700,activation='relu'))
model.add(Dense(units=700,activation='relu'))
model.add(Dense(units=10,activation='softmax'))
model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=100,epochs=50)

# with open('model.json','r') as f:
#         model = model_from_json(f.read())
#         f.close()
model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
# model.load_weights('model.h5')
score = model.evaluate(x_train,y_train)
print('\nTrain Acc:',score)
score = model.evaluate(x_test,y_test)
print('\nTest Acc:',score)
'''
model_json = model.to_json()
with open('model.json','w') as f:
        f.write(model_json)
        f.close()
model.save_weights('model.h5')
'''
