from create_training_file import create_training_wav, get_files
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D, Conv2D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape, Flatten
from keras.optimizers import Adam
import numpy as np
from pydub import AudioSegment
from create_training_file import graph_spectrogram
import matplotlib.pyplot as plt
import os




import keras
import tensorflow as tf


config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 12} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)


import pickle

def create_model(input_shape):
    # X_input = Input(shape=input_shape)
    #
    # # Convolutional layer
    # X = Conv1D(196, kernel_size=15, strides=4)(X_input)  # CONV1D
    # X = BatchNormalization()(X)  # Batch normalization
    # X = Activation('relu')(X)  # ReLu activation
    # X = Dropout(0.7)(X)
    #
    # # GRU layer
    # X = GRU(units=128, return_sequences=True)(X)  # GRU (use 128 units and return the sequences)
    # X = Dropout(0.7)(X)  # dropout (use 0.8)
    # X = BatchNormalization()(X)  # Batch normalization
    #
    # # GRU Layer
    # X = GRU(units=128, return_sequences=True)(X)  # GRU (use 128 units and return the sequences)
    # X = Dropout(0.7)(X)  # dropout (use 0.8)
    # X = BatchNormalization()(X)  # Batch normalization
    # X = Dropout(0.7)(X)  # dropout (use 0.8)
    #
    # # Time-distributed dense layer
    # X = TimeDistributed(Dense(1, activation="sigmoid", name="Output layer"))(X)  # time distributed  (sigmoid)
    #
    # model = Model(inputs=X_input, outputs=X)
    X_input = Input(shape=input_shape)

    # Step 1: CONV layer : for extracting features
    X = Conv1D(filters=256, kernel_size=15, strides=4)(X_input)  # CONV1D
    # X = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(X)  # Batch normalization
    # X = Activation('relu')(X)  # ReLu activation
    # X = Dropout(0.8)(X)  # dropout (use 0.8)

    # Step 2: First GRU Layer
    # X = GRU(units=128, return_sequences=True)(X)  # GRU (use 128 units and return the sequences)
    # X = Dropout(0.8)(X)  # dropout (use 0.8)
    # X = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(X)  # Batch normalization

    # Step 3: Second GRU Layer
    X = GRU(units=128, return_sequences=True)(X)  # GRU (use 128 units and return the sequences)
    # X = Dropout(0.8)(X)  # dropout (use 0.8)
    # X = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(X)  # Batch normalization
    # X = Dropout(0.8)(X)  # dropout (use 0.8)

    # Step 4: Time-distributed dense layer
    X = TimeDistributed(Dense(1, activation="sigmoid"))(X)  # time distributed  (sigmoid)

    model = Model(inputs=X_input, outputs=X)
    return model

noises_path = 'dataset/noise'
i = 0

Tx = 5511  # The number of time steps input to the model from the spectrogram
n_freq = 101  # Number of frequencies input to the model at each time step of the spectrogram

model = create_model(input_shape=(Tx, n_freq))
#opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01, amsgrad=False)
opt= Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])


def train():
    i = 0
    X_train=[]
    Y_train = []
    X_eval=[]
    Y_eval=[]
    for root, dirs, files in os.walk(noises_path):
        for file in files:
            if file.endswith("wav"):
                num=0
                while num<10:
                    path = os.path.join(root, file)
                    print("Taking noise: " + path)
                    sound = AudioSegment.from_wav(path)
                    x, y = create_training_wav(sound, "testing" + str(i)+"_"+str(num))

                    x=np.swapaxes(x, 0, 1)
                    print("LAAAAAAAAAAAAAAAA FORMIIIIIIIIIIIIIIIIITA:"+str(x.shape))
                    X_train.append(x)
                    #print(X_train)
                    y=np.swapaxes(y, 0, 1)
                    Y_train.append(y)

                    if num==5:
                        X_eval=X_train
                        Y_eval=Y_train

                    #X_train=np.array(X_train)
                    #print(X_train.shape)
                    # X_train = np.reshape(x, (1, x.shape[1], x.shape[0]))
                    # Y_train = np.reshape(y, (1, y.shape[1], y.shape[0]))

                    # X_train = np.expand_dims(X_train, axis=0)
                    # Y_train = np.expand_dims(Y_train, axis=0)

                    #print("SHAPE FOR FEEDING: "+str(X_train.shape), str(y.shape))
                    # try:
                    #     model.fit(X_train, y, epochs=4, batch_size=5)
                    # except:
                    #     pass
                    # # Mostramos el archivo y la posicion de los positivos
                    #print("Mostrando ejemplo de entrenamiento")
                    #graph_spectrogram('dataset/training_data/train_testing'+ str(i)+"_"+str(num)+'.wav', True)
                    #plt.plot(y)
                    #plt.show()
                    num+=1
                i += 1
    return np.array(X_train), np.array(Y_train), np.array(X_eval), np.array(Y_eval)
c=0
# try:
while c<20:
    X_train, Y_train, X_Eval, Y_Eval=train()

    print("KAKAKAKAKAkAKATUAS")
    print(X_train.shape, Y_train.shape)
    model.fit(X_train, Y_train, epochs=3, batch_size=5, validation_data=(X_Eval, Y_Eval))

    c+=1
scores = model.evaluate(X_Eval, Y_Eval, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
model.save("alex_test_model.h5")
# except Exception as e:
#     print("EEEEEEEEEEEEEEEEEERROR:  "+str(e))
# finally:
#
# print("Mostrando ejemplo de entrenamiento")
# graph_spectrogram('dataset/training_data/train_testing'+str(i-1)+'.wav')
# plt.plot(y[0])
# plt.show()
#
# print("LOADING X_Y_TRAIN")
# X = np.load("./XY_train/X.npy")
# Y = np.load("./XY_train/Y.npy")
# print(X.shape, Y.shape)
# for noise in noises:
#     x,y=create_training_wav(noise, "first_iter_"+str(i))
#     x_train.append(x)
#     y_train.append(y)
#     i+=1
# i=0
# for noise in noises:
#     x, y=create_training_wav(noise, "second_iter_"+str(i))
#     x_train.append(x)
#     y_train.append(y)
#     i+=1
#
# with open("X_training_data.pickle", "wb") as f:
#     pickle.dump(x_train[0], f)
# with open("Y_training_data.pickle", "wb") as f:
#     pickle.dump(y_train[0], f)
#




# with open("X_training_data.pickle", "rb") as f:
#     x_train = pickle.load(f)
# with open("Y_training_data.pickle", "rb") as f:
#     y_train = pickle.load(f)





# model=create_model(X_train.shape)
model.summary()






# model.save("alex_test_model.h5")
