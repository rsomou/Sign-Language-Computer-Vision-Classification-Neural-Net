# Sign-Language-Computer-Vision-Classification-Neural-Net
Computer Vision based on 87,000 images of sign language images to classify them into letters using a Convolutional Neural Net in Keras

Rajakrishnan Somou somou@usc.edu

The dataset used utilized around 3000 images of hand symbols ranging from 'A' to 'Z' and 3 special characters for a total of 29 different classes. In the data preprocessing, I utilized only around 1000 images out of each category to reduce the ammount of time and computational power spent on training. I appended each image directly from the directory a np array in which I resized them to 32x32, normalized each pixel value to 1-0 and converted to grayscale to optimize and reduce complexity of the data and make it better suited for training. For the labels, I took the name of the directory the file was contained in and then applied one hot encoding to the transformed labels to give a 29 dimensional vector that could be compared to the 29 dimensional output of the CNN. Then I split the training and testing data 1:9 to finalize the data transformation.

In the model, I utilized a 3 layered CNN with 2x2 Max pooling and batch normalization then which was then flatten and applied with dropout which preprocessed the input into a 3 layer dense neural net that took in the flattened feature vector and outputted a 29 dim vector. The Convolutional layers utilized a 3x3 kernal to create 64 filters to 128 to 256. Then to reduce dimensionality, max pooling layers were added after each Conv2D which selected the greatest value out of a 2x2 grid that was convolved on each filter map. And lastly, batch normalization was utilized to normalize the activation vectors from the Conv2D layers to then be passed into further Conv2D layers. This happened 3 times to create a 4x4x256 tensor output which was flattened to a 4096 dimensional vector abstracting features from the original transformed image. Dropout was applied after to reduce the chance of overfitting as it randomally dropped input nodes by 20% and increased the rest of the data by a scaled amount so the sum would be conserved. Then the flattened vector went through a 4096 to 1024 to 29 dense neural network using reLU as its activation in the first 2 layers to prevent gradient vanishing. And lasty, the final layer used a softmax activation to create a probability vector to then classify the image.

