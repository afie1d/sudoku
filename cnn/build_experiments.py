import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import copy

from sklearn.model_selection import train_test_split
import keras
from keras import regularizers
from keras.layers import Activation
from keras.layers import Conv2D, BatchNormalization, Dense, Flatten, Reshape, Dropout

df = pd.read_csv('sudoku/sudoku.csv')
df.head()

que = df['quizzes'].values
soln = df['solutions']

#prepocessing data
feat = []
label = []

for i in que:

    x = np.array([int(j) for j in i]).reshape((9,9,1))
    feat.append(x)

feat = np.array(feat)
feat = feat/9
feat -= .5    

for i in soln:

    x = np.array([int(j) for j in i]).reshape((81,1)) - 1
    label.append(x)   

label = np.array(label)

#splitting data
X_train, X_test, y_train, y_test = train_test_split(feat,label, test_size=0.33, random_state=42)


#creating model #1
# 5 Convolution layers (regularized), 2 fully connected layers
def get_model1():
    
    model = keras.models.Sequential()

    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', input_shape=(9,9,1)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(1,1), activation='relu', padding='same'))

    model.add(Flatten())
    model.add(Dense(81*9))
    model.add(Flatten())
    model.add(Reshape((81, 9)))
    model.add(Dense(81))
    model.add(Activation('softmax'))
    
    return model

model1 = get_model1()





#creating model #2
# 11 convolutional layers + dropout (regularized), one fully-connected layer
def get_model2():
    
    model = keras.models.Sequential()

    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', input_shape=(9,9,1)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(1,1), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(1,1), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(1,1), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(81*9))
    model.add(Reshape((-1, 9)))
    model.add(Activation('softmax'))
    
    return model

model2 = get_model2()


#creating model #3
# 15 convolutional layers, one fully-connected layer
def get_model3():
    
    model = keras.models.Sequential()

    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', input_shape=(9,9,1)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(1,1), activation='relu', padding='same'))

    model.add(Flatten())
    model.add(Dense(81*9))
    model.add(Reshape((-1, 9)))
    model.add(Activation('softmax'))
    
    return model

model3 = get_model3()

#Train models
adam = keras.optimizers.Adam(lr=.001)

model1.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model1.fit(X_train, y_train, batch_size=32, epochs=2)
model1.save('experimentModel1.keras')

model2.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model2.fit(X_train, y_train, batch_size=32, epochs=2)
model2.save('experimentModel2.keras')

model3.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model3.fit(X_train, y_train, batch_size=32, epochs=2)
model3.save('experimentModel3.keras')

print("Training complete. Beginning testing...\n")


# Test models

model1_loss, model1_accuracy = model1.evaluate(X_test, y_test)
model2_loss, model2_accuracy = model2.evaluate(X_test, y_test)
model3_loss, model3_accuracy = model3.evaluate(X_test, y_test)

print("Testing Complete. \n Model 1 accuracy: ", model1_accuracy, "\nModel 2 accuracy: ", model2_accuracy, "\nModel 3 accuracy: ", model3_accuracy)