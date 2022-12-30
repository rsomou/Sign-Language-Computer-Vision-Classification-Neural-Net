# Sign-Language-Computer-Vision-Classification-Neural-Net
Computer Vision based on 87,000 images of sign language images to classify them into letters using a Convolutional Neural Net in Keras

Rajakrishnan Somou somou@usc.edu

The dataset used utilized around 3000 images of hand symbols ranging from 'A' to 'Z' and 3 special characters for a total of 29 different classes. In the data preprocessing, I utilized only around 1000 images out of each category to reduce the ammount of time and computational power spent on training. I appended each image directly from the directory a np array in which I resized them to 32x32, normalized each pixel value to 1-0 and converted to grayscale to optimize and reduce complexity of the data and make it better suited for training. For the labels, I took the name of the directory the file was contained in and then applied one hot encoding to the transformed labels to give a 29 dimensional vector that could be compared to the 29 dimensional output of the CNN. Then I split the training and testing data 1:9 to finalize the data transformation.

In the model, I utilized a 3 layered CNN with 2x2 Max pooling and batch normalization then which was then flatten and applied with dropout which preprocessed the input into a 3 layer dense neural net that took in the flattened feature vector and outputted a 29 dim vector. The Convolutional layers utilized a 3x3 kernal to create 64 filters to 128 to 256. Then to reduce dimensionality, max pooling layers were added after each Conv2D which selected the greatest value out of a 2x2 grid that was convolved on each filter map. And lastly, batch normalization was utilized to normalize the activation vectors from the Conv2D layers to then be passed into further Conv2D layers. This happened 3 times to create a 4x4x256 tensor output which was flattened to a 4096 dimensional vector abstracting features from the original transformed image. Dropout was applied after to reduce the chance of overfitting as it randomally dropped input nodes by 20% and increased the rest of the data by a scaled amount so the sum would be conserved. Then the flattened vector went through a 4096 to 1024 to 29 dense neural network using reLU as its activation in the first 2 layers to prevent gradient vanishing. And lasty, the final layer used a softmax activation to create a probability vector to then classify the image.

Output:

(29000, 32, 32, 1)
(29000, 29)
Training data: (26100, 32, 32, 1)
Test data: (2900, 32, 32, 1)
Model: "Subby"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_55 (Conv2D)           (None, 32, 32, 64)        640       
_________________________________________________________________
max_pooling2d_55 (MaxPooling (None, 16, 16, 64)        0         
_________________________________________________________________
batch_normalization_55 (Batc (None, 16, 16, 64)        256       
_________________________________________________________________
conv2d_56 (Conv2D)           (None, 16, 16, 128)       73856     
_________________________________________________________________
max_pooling2d_56 (MaxPooling (None, 8, 8, 128)         0         
_________________________________________________________________
batch_normalization_56 (Batc (None, 8, 8, 128)         512       
_________________________________________________________________
conv2d_57 (Conv2D)           (None, 8, 8, 256)         295168    
_________________________________________________________________
max_pooling2d_57 (MaxPooling (None, 4, 4, 256)         0         
_________________________________________________________________
batch_normalization_57 (Batc (None, 4, 4, 256)         1024      
_________________________________________________________________
conv2d_58 (Conv2D)           (None, 4, 4, 512)         1180160   
_________________________________________________________________
max_pooling2d_58 (MaxPooling (None, 2, 2, 512)         0         
_________________________________________________________________
batch_normalization_58 (Batc (None, 2, 2, 512)         2048      
_________________________________________________________________
flatten_14 (Flatten)         (None, 2048)              0         
_________________________________________________________________
dropout_15 (Dropout)         (None, 2048)              0         
_________________________________________________________________
dense_39 (Dense)             (None, 2048)              4196352   
_________________________________________________________________
dense_40 (Dense)             (None, 512)               1049088   
_________________________________________________________________
dense_41 (Dense)             (None, 29)                14877     
=================================================================
Total params: 6,813,981
Trainable params: 6,812,061
Non-trainable params: 1,920
_________________________________________________________________
Epoch 1/5
653/653 [==============================] - 107s 162ms/step - loss: 0.4122 - accuracy: 0.8747 - val_loss: 0.8959 - val_accuracy: 0.7864

Epoch 00001: saving model to training_1/cp.ckpt
Epoch 2/5
653/653 [==============================] - 102s 156ms/step - loss: 0.1161 - accuracy: 0.9686 - val_loss: 20.0782 - val_accuracy: 0.2588

Epoch 00002: saving model to training_1/cp.ckpt
Epoch 3/5
653/653 [==============================] - 102s 156ms/step - loss: 0.0741 - accuracy: 0.9811 - val_loss: 0.0885 - val_accuracy: 0.9728

Epoch 00003: saving model to training_1/cp.ckpt
Epoch 4/5
653/653 [==============================] - 102s 157ms/step - loss: 0.0722 - accuracy: 0.9834 - val_loss: 0.1901 - val_accuracy: 0.9617

Epoch 00004: saving model to training_1/cp.ckpt
Epoch 5/5
653/653 [==============================] - 102s 156ms/step - loss: 0.0356 - accuracy: 0.9918 - val_loss: 0.0034 - val_accuracy: 0.9989

Epoch 00005: saving model to training_1/cp.ckpt
91/91 [==============================] - 4s 46ms/step - loss: 0.0024 - accuracy: 0.9997
Test accuracy: 0.9996551871299744
Test loss: 0.00244730687700212
              precision    recall  f1-score   support

           A       1.00      1.00      1.00       108
           B       1.00      1.00      1.00        95
           C       1.00      1.00      1.00        98
           D       1.00      1.00      1.00        92
           E       1.00      1.00      1.00        91
           F       1.00      1.00      1.00       118
           G       1.00      1.00      1.00       111
           H       1.00      1.00      1.00       106
           I       1.00      1.00      1.00       108
           J       1.00      1.00      1.00        86
           K       1.00      1.00      1.00       106
           L       1.00      1.00      1.00        92
           M       1.00      1.00      1.00       118
           N       1.00      1.00      1.00       101
           O       1.00      1.00      1.00       101
           P       1.00      0.99      1.00       108
           Q       1.00      1.00      1.00        80
           R       1.00      1.00      1.00        93
           S       1.00      1.00      1.00        97
           T       1.00      1.00      1.00        99
           U       1.00      1.00      1.00       121
           V       1.00      1.00      1.00       100
           W       1.00      1.00      1.00       101
           X       1.00      1.00      1.00        89
           Y       0.99      1.00      1.00       102
           Z       1.00      1.00      1.00        95
         del       1.00      1.00      1.00        87
     nothing       1.00      1.00      1.00        96
       space       1.00      1.00      1.00       101

   micro avg       1.00      1.00      1.00      2900
   macro avg       1.00      1.00      1.00      2900
weighted avg       1.00      1.00      1.00      2900
 samples avg       1.00      1.00      1.00      2900
