from create_training_file import create_training_wav, get_files
import numpy as np

import pickle
noises=get_files('dataset/noise')
x_train=[]
y_train=[]
i=0
for noise in noises:
    x,y=create_training_wav(noise, "first_iter_"+str(i))
    x_train.append(x)
    y_train.append(y)
    i+=1
i=0
for noise in noises:
    x, y=create_training_wav(noise, "second_iter_"+str(i))
    x_train.append(x)
    y_train.append(y)
    i+=1
print(x_train)
print(y_train)
with open("X_training_data.pickle", "wb") as f:
    pickle.dump(x_train[0], f)
with open("Y_training_data.pickle", "wb") as f:
    pickle.dump(y_train[0], f)
