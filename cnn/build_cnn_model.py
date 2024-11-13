import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import copy

from sklearn.model_selection import train_test_split
import keras
from keras.layers import Activation
from keras.layers import Conv2D, BatchNormalization, Dense, Flatten, Reshape

df = pd.read_csv('sudoku/sudoku/sudoku.csv')
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

#creating model
def get_model():
    
    model = keras.models.Sequential()

    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', input_shape=(9,9,1)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(1,1), activation='relu', padding='same'))

    model.add(Flatten())
    model.add(Dense(81*9))
    model.add(Reshape((-1, 9)))
    model.add(Activation('softmax'))
    
    return model

model = get_model()

#training our model
adam = keras.optimizers.Adam(lr=.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=adam)

model.fit(X_train, y_train, batch_size=32, epochs=2)


model.save('solverModel.keras')