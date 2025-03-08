# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import cv2
import random as rn
import matplotlib.pyplot as plt
from matplotlib import style
from tensorflow import keras
from keras.layers import Input, Lambda, Dense, Flatten, Dropout, BatchNormalization
from keras.models import Model, Sequential
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from keras.layers.convolutional import Conv2D, MaxPooling2D
# from keras.layers import Dense, Flatten,
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

from google.colab import drive
drive.mount('/content/drive')

"""# **Preparing the Dataset**"""

imagepaths = []

import os
for dirname, _, filenames in os.walk('/content/drive/MyDrive/final-dataset'):
    for filename in filenames:
        path = os.path.join(dirname, filename)
        imagepaths.append(path)

print(len(imagepaths))

IMG_SIZE=128
X=[]
y=[]
for image in imagepaths:
    try:
        img = cv2.imread(image,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        X.append(np.array(img))
        if(image.startswith('/content/drive/MyDrive/final-dataset/good/')):
            y.append('good')
        else:
            y.append('bad')
    except:
        pass

fig,ax=plt.subplots(2,5)
plt.subplots_adjust(bottom=0.3, top=0.7, hspace=0)
fig.set_size_inches(15,15)

for i in range(2):
    for j in range (5):
        l=rn.randint(0,len(y))
        ax[i,j].imshow(X[l][:,:,::-1])
        ax[i,j].set_title(y[l])
        ax[i,j].set_aspect('equal')

"""# **Data Engineering**
# **Label Encoding the Y array (i.e. Negative->0, Positive->1) & then One Hot Encoding**
"""

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

le=LabelEncoder()
Y=le.fit_transform(y)
Y=to_categorical(Y,2)
X=np.array(X)

print(Y)

"""# **Splitting into Training and Testing Sets**"""

x_train, x_test, y_train, y_test = train_test_split (X,Y,test_size=0.30,random_state=0)

x_train.shape

# Normalize Images

x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis=1)

"""# **Data Augmentation to prevent Overfitting**"""

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

"""# **Building the CNN Model**"""

model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

x_train.shape

x_test.shape

y_train.shape

y_test.shape

# custom_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
import tensorflow as tf
model.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())



# history = model.fit(x_train, y_train, epochs=15, batch_size=12, verbose=2, validation_data=(x_test, y_test))
batch_size=30
epochs=15
history = model.fit(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs,
                              validation_data = (x_test,y_test), verbose = 2)

loss, accuracy = model.evaluate(x_test, y_test)
print('Test accuracy: {:2.2f}%'.format(accuracy*100))

prediction = model.predict(x_test)

prediction

y_pred = np.argmax(prediction, axis=1)

model.save('model.h5')

y_pred

history.history.keys()

plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('Accuracy and Loss Plot')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['accuracy','loss'], loc='upper right')
plt.show()

plt.plot(history.history['val_accuracy'])
plt.plot(history.history['val_loss'])
plt.title('Testing Accuracy and Loss Plot')
plt.xlabel('epochs')
plt.ylabel('Validation accuracy and loss')
plt.legend(['val_accuracy','val_loss'], loc='upper left')
plt.show()

#testing the model for confusion matrix
y_test1=y_test.astype(int)
y_test2=[]
for i in y_test1:
    a=1
    if(i[0]==1 and i[1]==0):
        a=0
    y_test2.append(a)

pd.DataFrame(confusion_matrix(y_test2, y_pred),
             columns=["Predicted NEGATIVE", "Predicted POSITIVE"],
             index=["Actual NEGATIVE", "Actual POSITIVE"])

import seaborn as sns
cmx = confusion_matrix(y_test2, y_pred)
# Adjust the size of the figure
plt.figure(figsize=(7, 6))
sns.heatmap(cmx, annot=True)

from sklearn.metrics import classification_report
print(classification_report(np.argmax(y_test, axis = 1),y_pred))

"""# **checking if the model recognize properly**"""

import os
folder_dir = '/content/drive/MyDrive/final-dataset'
categories = np.sort(os.listdir(folder_dir))
fig, ax = plt.subplots(6,6, figsize=(25, 40))

for i in range(6):
    for j in range(6):
        k = int(np.random.random_sample() * len(x_test))
        if(categories[np.argmax(y_test[k])] == categories[np.argmax(model.predict(x_test)[k])]):
            ax[i,j].set_title("TRUE: " + categories[np.argmax(y_test[k])], color='green')
            ax[i,j].set_xlabel("PREDICTED: " + categories[np.argmax(model.predict(x_test)[k])], color='green')
            ax[i,j].imshow(np.array(x_test)[k].reshape(IMG_SIZE, IMG_SIZE, 3), cmap='gray')
        else:
            ax[i,j].set_title("TRUE: " + categories[np.argmax(y_test[k])], color='red')
            ax[i,j].set_xlabel("PREDICTED: " + categories[np.argmax(model.predict(x_test)[k])], color='red')
            ax[i,j].imshow(np.array(x_test)[k].reshape(IMG_SIZE, IMG_SIZE, 3), cmap='gray')
