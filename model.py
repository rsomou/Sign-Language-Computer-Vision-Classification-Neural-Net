import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

train_dir = '../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train'
test_dir = '../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test'

# Gather data

images=[]
labels=[]

# Images in dataset range from 1-3000 (allows for selective use of total ammount of data within training and testing)

dataNum=1000

for root in os.listdir(train_dir):
    for index in range(1,dataNum):
        imagename=root+str(index)+'.jpg'
        img=cv2.imread(train_dir+'/'+root+'/'+imagename,1)
        img = cv2.resize(img, (32, 32))
        images.append(img)
        labels.append(root)

# One Hot Encoding Classes

encodedlabels=np.zeros(shape=(len(labels),29))
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
           'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
           'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

for index, element in enumerate(labels):
    encodedlabels[index][classes.index(element)]=1
    

# Normalize values of images to between 0-1

images=np.array(images)
images = images.astype('float32') 
images/=255

print(images.shape)
print(encodedlabels.shape)

#Split Testing and Training Data

x_train,x_test,y_train,y_test=train_test_split(images,encodedlabels,random_state=104,test_size=0.1)
print("Training data:", x_train.shape)
print("Test data:", x_test.shape)

# Creation and layering of Convolutional layers with Pooling and Dense Neural layers to classify

model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same', input_shape = (32, 32, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), padding='same', input_shape = (32, 32, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), padding='same', input_shape = (32, 32, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dense(29, activation='softmax'))

# Prints Model Architecture

model.summary()

adam = Adam(learning_rate=0.001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# Path of Checkpoint
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights and structure
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Training
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.2, callbacks=[cp_callback], shuffle = True, verbose=1)

# Evaluation
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)
