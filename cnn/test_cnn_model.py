import keras
import numpy as np
import pandas as pd
import copy
from sklearn.model_selection import train_test_split

"""This file will load the keras model and 
allow you to run a sudoku game on it to test"""

model = keras.models.load_model('sudoku/solverModel.keras')

def denorm(a):
    
    return (a+.5)*9

def norm(a):
    
    return (a/9)-.5

def inference_sudoku(sample):
    
    '''
        This function solve the sudoku by filling blank positions one by one.
    '''
    
    feat = copy.copy(sample)
    
    while(1):
        
        #predicting values
        out = model.predict(feat.reshape((1,9,9,1))) 
        out = out.squeeze()

        #getting predicted values
        pred = np.argmax(out, axis=1).reshape((9,9))+1 
        #getting probablity of each values
        prob = np.around(np.max(out, axis=1).reshape((9,9)), 2) 
        
        #creating mask for blank values
        feat = denorm(feat).reshape((9,9))
        #i.e it will give you a 2d array with True/1 and False/0 where 0 is found and where 0 is not found respectively
        mask = (feat==0)
     
        #if there are no 0 values left then break
        if(mask.sum()==0):
            break
            
        #getting probablities of values where 0 is present that is our blank values we need to fill
        prob_new = prob*mask
    
        #getting highest probablity index
        ind = np.argmax(prob_new)
        #getting row and col 
        x, y = (ind//9), (ind%9)
        
        #getting predicted value at that cell
        val = pred[x][y]
        #assigning that value
        feat[x][y] = val
        #again passing this sudoku with one value added to model to get next most confident value
        feat = norm(feat)
    
    return pred

def solve_sudoku(game):
    
    game = game.replace('\n', '')
    game = game.replace(' ', '')
    game = np.array([int(j) for j in game]).reshape((9,9,1))
    game = norm(game)
    game = inference_sudoku(game)
    return game


#TODO: Try changing this to another 9x9 game and see if it solves it!
game = '''
      3 0 0 2 0 0 0 0 0
0 0 0 1 0 7 0 0 0
7 0 6 0 3 0 5 0 0
0 7 0 0 0 9 0 8 0
9 0 0 0 2 0 0 0 4
0 1 0 8 0 0 0 5 0
0 0 9 0 4 0 3 0 1
0 0 0 7 0 2 0 0 0
0 0 0 0 0 8 0 0 6
      '''

game = solve_sudoku(game)

for i in game:
    print(i)


def valid(game):
    # Check rows
    for row in game:
        if not is_valid_set(row):
            return False
    
    # Check columns
    for col in range(9):
        column = [game[row][col] for row in range(9)]
        if not is_valid_set(column):
            return False
    
    # Check subgrids
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            subgrid = [game[row][col] for row in range(i, i + 3) for col in range(j, j + 3)]
            if not is_valid_set(subgrid):
                return False
    
    return True

def is_valid_set(nums):
    seen = set()
    for num in nums: 
        if num in seen:
            return False
        seen.add(num)
    return True

if valid(game):
    print("Valid Solution\n")
else: 
    print("Invalid Solution\n")



# Testing on data set ------------------------------------------------

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
adam = keras.optimizers.Adam(lr=.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

train_loss, train_accuracy = model.evaluate(X_train, y_train)
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Testing Complete. \n Training Accuracy: ", train_accuracy, "\n Test Accuracy: ", test_accuracy)