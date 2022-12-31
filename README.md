# Sign-Language-Computer-Vision-Classification-Neural-Net
Computer Vision based on 87,000 images of sign language images to classify them into letters using a Convolutional Neural Net in Keras

Rajakrishnan Somou somou@usc.edu

The dataset used utilized around 3000 images of hand symbols ranging from 'A' to 'Z' and 3 special characters for a total of 29 different classes. In the data preprocessing, I utilized only around 1000 images out of each category to reduce the ammount of time and computational power spent on training. I appended each image directly from the directory a np array in which I resized them to 32x32, normalized each pixel value to 1-0 and converted to grayscale to optimize and reduce complexity of the data and make it better suited for training. For the labels, I took the name of the directory the file was contained in and then applied one hot encoding to the transformed labels to give a 29 dimensional vector that could be compared to the 29 dimensional output of the CNN. Then I split the training and testing data 1:9 to finalize the data transformation.

In the model, I utilized a 4 layered CNN with 2x2 Max pooling and batch normalization then which was then flatten and applied with dropout which preprocessed the input into a 3 layer dense neural net that took in the flattened feature vector and outputted a 29 dim vector. The Convolutional layers utilized a 3x3 kernal to create 64 filters to 128 to 256 to 512. Then to reduce dimensionality, max pooling layers were added after each Conv2D which selected the greatest value out of a 2x2 grid that was convolved on each filter map. And lastly, batch normalization was utilized to normalize the activation vectors from the Conv2D layers to then be passed into further Conv2D layers. This happened 3 times to create a 2x2x512 tensor output which was flattened to a 2048 dimensional vector abstracting features from the original transformed image. Dropout was applied after to reduce the chance of overfitting as it randomally dropped input nodes by 20% and increased the rest of the data by a scaled amount so the sum would be conserved. Then the flattened vector went through a 2048 to 512 to 29 dense neural network using reLU as its activation in the first 2 layers to prevent gradient vanishing. And lasty, the final layer used a softmax activation to create a probability vector to then classify the image. For training, I varied around with the amount of epochs to prevent overfitting as well as the structure of the model to reduce the paramteres with in the training. for the final output, I trained with 3 epochs and with a batch_size of 32.

In terms of metrics, I utilized the classifcation report to see the precision, f1 score and accuracy of each class for the model. The total accuracy for the test split of the data was 93%. This was the result of the report given.

                 precision  recall   f1-score

           A       1.00      1.00      1.00      
           B       1.00      0.69      0.82        
           C       0.84      1.00      0.92        
           D       0.80      0.85      0.83        
           E       0.83      1.00      0.91        
           F       1.00      0.97      0.99       
           G       0.99      1.00      1.00       
           H       0.77      1.00      0.87       
           I       0.75      0.99      0.85       
           J       1.00      0.70      0.82       
           K       1.00      0.63      0.77       
           L       1.00      0.84      0.91        
           M       1.00      0.86      0.92       
           N       0.92      0.96      0.94       
           O       0.90      0.98      0.94       
           P       1.00      0.99      1.00       
           Q       0.94      1.00      0.97        
           R       0.97      0.40      0.56        
           S       1.00      0.71      0.83        
           T       0.74      1.00      0.85        
           U       0.76      0.97      0.85       
           V       0.98      0.81      0.89       
           W       1.00      0.75      0.86       
           X       0.49      1.00      0.66        
           Y       1.00      0.99      1.00       
           Z       1.00      0.78      0.88        
         del       1.00      1.00      1.00        
     nothing       0.97      1.00      0.98        
       space       1.00      0.86      0.93       
 
 I believe that this specific model does very well in terms of not overfitting to the data but as well as correctly predicting new data. Changing the layers and overall structure lead to significant changes in training time and accuracy at the end of prediction. With 4 epoches I achieved a 99% accuracy, but I thought this was due to overfitting so with just 3 epochs this model could still achieve a 93% accuracy which shows the versatitalty of the model in training as well as testing. This model can be a very useful tool in diplomatic settings or even day to day interactions as the immediate translation of sign language to people that dont know the symbols within ASL could have those conversations readily. If I were to continue this project, maybe implementing this into a camera of some sort would be the next logical step to to make it usable and accessible to the real world. In terms of code, this would drastically make data input much more complex as taking a live feed would have alot of data preprocessing steps but at the end of the day it definitely seems feasible to accomplish.
