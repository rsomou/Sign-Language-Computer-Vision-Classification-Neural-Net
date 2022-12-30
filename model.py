import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils
from tensorflow.keras import layers, models, optimizers

total_dir= '../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train'


# Gather data

images=[]
labels=[]

# Images in dataset range from 1-3000 (allows for selective use of total ammount of data within training and testing)

dataNum=1000

for root in os.listdir(total_dir):
    for index in range(1,dataNum):
        imagename=root+str(index)+'.jpg'
        img=cv2.imread(total_dir+'/'+root+'/'+imagename,1)
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

# Build model
def create_model():

    # Add layers of model
    
    conv_layer_1 = layers.Conv2D(64, (3, 3), padding='same', input_shape = (32, 32, 3), activation='relu')
    conv_layer_2 = layers.Conv2D(128, (3, 3), padding='same', input_shape = (32, 32, 3), activation='relu')
    conv_layer_3 = layers.Conv2D(256, (3, 3), padding='same', input_shape = (32, 32, 3), activation='relu') 
    pooling_layer_1 = layers.MaxPooling2D(pool_size=(2, 2))
    pooling_layer_2 = layers.MaxPooling2D(pool_size=(2, 2))
    pooling_layer_3 = layers.MaxPooling2D(pool_size=(2, 2))
    batch_norm_layer_1 = layers.BatchNormalization() 
    batch_norm_layer_2 = layers.BatchNormalization()
    batch_norm_layer_3 = layers.BatchNormalization() 
    flatten_layer = layers.Flatten()
    dropout = layers.Dropout(0.2)
    dense_layer = layers.Dense(1024, activation='relu')
    output_layer = layers.Dense(29, activation='softmax')
    
    # Creation of Model
                                
    model = models.Sequential([
        conv_layer_1, 
        pooling_layer_1,
        batch_norm_layer_1,
        conv_layer_2, 
        pooling_layer_2,
        batch_norm_layer_2,
        conv_layer_3, 
        pooling_layer_3,
        batch_norm_layer_3,
        flatten_layer,
        dropout,
        dense_layer,
        output_layer
    ])
    
    # Compile model
    # Categorical crossentropy is used since the data labels are categorical and one hot encoded
    # Adam optimiser is used as it applies both momentum and RMSP(Root Mean Squared Propagation)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

# Prints and Creates Model Architecture

model = create_model()
model.summary()

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
