
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot # plotly offline mode
init_notebook_mode(connected=True) 
import os

x = list() #create x data
y = list() # create y data
for i in os.listdir("C:\\Users\\Dua Fatima\\Downloads\\cell-images-for-detecting-malaria\\cell_images\\Parasitized\\"): #read all parasitized data 
    if ".png" in i: #this if block for only read .png files
        path = "C:\\Users\\Dua Fatima\\Downloads\\cell-images-for-detecting-malaria\\cell_images\\Parasitized\\"+i # create path
        img = plt.imread(path) # and read created path
        img = cv2.resize(img,(40,40)) # resize image for lower processing power
        x.append(img) # append image to x data
        y.append(1) 
for i in os.listdir("C:\\Users\\Dua Fatima\\Downloads\\cell-images-for-detecting-malaria\\cell_images\\Uninfected\\"):
    if ".png" in i:
        path = "C:\\Users\\Dua Fatima\Downloads\\cell-images-for-detecting-malaria\\cell_images\\Uninfected\\"+i
        img = plt.imread(path)
        img = cv2.resize(img,(40,40))
        x.append(img)
        y.append(0)
x = np.array(x)  

#reshapeing data
x = x.reshape(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3])
x = x**8

# i used sklearn module for splitting process
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

#import LogisticRegression and fit with out datas
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="lbfgs")
lr.fit(x_train,y_train)

# Test my Logistic Regression Model
print("Logistic Regression Accuracy : {0:.2f}%".format(100 * lr.score(x_test,y_test)))

#import modules i will use
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential # initialize neural network library
from keras.layers import Dense # build our layers library
# build our model
def build_classifier():
    classifier = Sequential() # initialize neural network
    classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1])) # firt hidden layer 
    classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')) # last layer
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier # return our model

model = build_classifier()
model.summary()

classifier = KerasClassifier(build_fn = build_classifier, epochs = 100)
history = classifier.fit(x_train,y_train)

plt.plot(history.history['acc'])
plt.plot(history.history['loss'])
plt.title('Model Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy/Loss')
plt.legend(['Accuracy', 'Loss'], loc='upper left')

plt.show()

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#Prepared the data and implemented this data to Logistic Regression and ANN
#Firstly implemented Logistic Regression and this accuracy returned to 66.15%
#Secondly implemented ANN and this accuracy 97% while training but during testing this value dropping the 71%
