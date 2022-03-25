In this challenge, you must use what you've learned to train a convolutional neural network model that classifies images of animals you might find on a safari adventure.

*(This is an exercise from [a course on basic machine learning](https://github.com/MicrosoftDocs/ml-basics) sponsored by Microsoft)*

## Explore the data

The training images you must use are in the **/safari/training** folder. Run the cell below to see an example of each image class, and note the shape of the images (which indicates the dimensions of the image and its color channels).


```python
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

# The images are in the data/shapes folder
data_path = 'data/safari/training'

# Get the class names
classes = os.listdir(data_path)
classes.sort()
print(len(classes), 'classes:')
print(classes)

# Show the first image in each folder
fig = plt.figure(figsize=(12, 12))
i = 0
for sub_dir in os.listdir(data_path):
    i+=1
    img_file = os.listdir(os.path.join(data_path,sub_dir))[0]
    img_path = os.path.join(data_path, sub_dir, img_file)
    img = mpimg.imread(img_path)
    img_shape = np.array(img).shape
    a=fig.add_subplot(1, len(classes),i)
    a.axis('off')
    imgplot = plt.imshow(img)
    a.set_title(img_file + ' : ' + str(img_shape))
plt.show()
```

    4 classes:
    ['elephant', 'giraffe', 'lion', 'zebra']
    


    
![png]({{ site.baseurl }}/assets/img/2022-03-24-safari-cnn-challenge_files/2022-03-24-safari-cnn-challenge_13_0.png)
    


Now that you've seen the images, use your preferred framework (PyTorch or TensorFlow) to train a CNN classifier for them. Your goal is to train a classifier with a validation accuracy of 95% or higher.

Add cells as needed to create your solution.

> **Note**: There is no single "correct" solution. Sample solutions are provided in [05 - Safari CNN Solution (PyTorch).ipynb](05%20-%20Safari%20CNN%20Solution%20(PyTorch).ipynb) and [05 - Safari CNN Solution (TensorFlow).ipynb](05%20-%20Safari%20CNN%20Solution%20(TensorFlow).ipynb).


```python
# Your Code to train a CNN model...
import tensorflow
from  tensorflow import keras
print('TensorFlow version:',tensorflow.__version__)
print('Keras version:',keras.__version__)
```

    TensorFlow version: 2.8.0
    Keras version: 2.8.0
    


```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_size = (200, 200)
batch_size = 30

print("Getting Data...")
datagen = ImageDataGenerator(rescale=1./255, # normalize pixel values
                             validation_split=0.3) # hold back 30% of the images for validation

print("Preparing training dataset...")
train_generator = datagen.flow_from_directory(
    data_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training') # set as training data

print("Preparing validation dataset...")
validation_generator = datagen.flow_from_directory(
    data_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation') # set as validation data

classnames = list(train_generator.class_indices.keys())
print('Data generators ready')
```

    Getting Data...
    Preparing training dataset...
    Found 280 images belonging to 4 classes.
    Preparing validation dataset...
    Found 116 images belonging to 4 classes.
    Data generators ready
    

Extra reading - [choosing kernal size](https://medium.com/analytics-vidhya/how-to-choose-the-size-of-the-convolution-filter-or-kernel-size-for-cnn-86a55a1e2d15)


```python
# Define a CNN classifier network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# Define the model as a sequence of layers
model = Sequential()

# The input layer accepts an image and applies a convolution that uses 32 6x6 filters and a rectified linear unit activation function
model.add(Conv2D(32, (6, 6), input_shape=train_generator.image_shape, activation='relu'))

# Next we'll add a max pooling layer with a 2x2 patch
model.add(MaxPooling2D(pool_size=(2,2)))

# We can add as many layers as we think necessary - here we'll add another convolution and max pooling layer
model.add(Conv2D(32, (6, 6), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# And another set
model.add(Conv2D(32, (6, 6), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# A dropout layer randomly drops some nodes to reduce inter-dependencies (which can cause over-fitting)
model.add(Dropout(0.2))

# Flatten the feature maps 
model.add(Flatten())

# Generate a fully-connected output layer with a predicted probability for each class
# (softmax ensures all probabilities sum to 1)
model.add(Dense(train_generator.num_classes, activation='softmax'))

# With the layers defined, we can now compile the model for categorical (multi-class) classification
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d (Conv2D)             (None, 195, 195, 32)      3488      
                                                                     
     max_pooling2d (MaxPooling2D  (None, 97, 97, 32)       0         
     )                                                               
                                                                     
     conv2d_1 (Conv2D)           (None, 92, 92, 32)        36896     
                                                                     
     max_pooling2d_1 (MaxPooling  (None, 46, 46, 32)       0         
     2D)                                                             
                                                                     
     conv2d_2 (Conv2D)           (None, 41, 41, 32)        36896     
                                                                     
     max_pooling2d_2 (MaxPooling  (None, 20, 20, 32)       0         
     2D)                                                             
                                                                     
     dropout (Dropout)           (None, 20, 20, 32)        0         
                                                                     
     flatten (Flatten)           (None, 12800)             0         
                                                                     
     dense (Dense)               (None, 4)                 51204     
                                                                     
    =================================================================
    Total params: 128,484
    Trainable params: 128,484
    Non-trainable params: 0
    _________________________________________________________________
    None
    


```python
# Train the model over 5 epochs using 30-image batches and using the validation holdout dataset for validation

num_epochs = 5
history = model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // batch_size,
    epochs = num_epochs)
```

    Epoch 1/5
    9/9 [==============================] - 65s 8s/step - loss: 1.4565 - accuracy: 0.2960 - val_loss: 1.2468 - val_accuracy: 0.4556
    Epoch 2/5
    9/9 [==============================] - 49s 6s/step - loss: 0.9356 - accuracy: 0.6120 - val_loss: 0.7247 - val_accuracy: 0.5000
    Epoch 3/5
    9/9 [==============================] - 51s 6s/step - loss: 0.4794 - accuracy: 0.7920 - val_loss: 0.2357 - val_accuracy: 0.9111
    Epoch 4/5
    9/9 [==============================] - 47s 6s/step - loss: 0.1650 - accuracy: 0.9520 - val_loss: 0.0352 - val_accuracy: 1.0000
    Epoch 5/5
    9/9 [==============================] - 48s 5s/step - loss: 0.0268 - accuracy: 1.0000 - val_loss: 0.0109 - val_accuracy: 1.0000
    


```python
%matplotlib inline
from matplotlib import pyplot as plt

epoch_nums = range(1,num_epochs+1)
training_loss = history.history["loss"]
validation_loss = history.history["val_loss"]
plt.plot(epoch_nums, training_loss)
plt.plot(epoch_nums, validation_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()
```


    
![png]({{ site.baseurl }}/assets/img/2022-03-24-safari-cnn-challenge_files/2022-03-24-safari-cnn-challenge_8_0.png)
    



```python
# Tensorflow doesn't have a built-in confusion matrix metric, so we'll use SciKit-Learn
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
%matplotlib inline

print("Generating predictions from validation data...")
# Get the image and label arrays for the first batch of validation data
x_test = validation_generator[0][0]
y_test = validation_generator[0][1]

# Use the model to predict the class
class_probabilities = model.predict(x_test)

# The model returns a probability value for each class
# The one with the highest probability is the predicted class
predictions = np.argmax(class_probabilities, axis=1)

# The actual labels are hot encoded (e.g. [0 1 0], so get the one with the value 1
true_labels = np.argmax(y_test, axis=1)

# Plot the confusion matrix
cm = confusion_matrix(true_labels, predictions)
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(classnames))
plt.xticks(tick_marks, classnames, rotation=0)
plt.yticks(tick_marks, classnames)
plt.xlabel("Predicted Shape")
plt.ylabel("Actual Shape")
plt.show()
```

    Generating predictions from validation data...
    


    
![png]({{ site.baseurl }}/assets/img/2022-03-24-safari-cnn-challenge_files/2022-03-24-safari-cnn-challenge_9_1.png)
    


## Save your model

Add code below to save your model's trained weights.


```python
# Code to save your model
modelFileName = 'models/animal_classifier.h5'
model.save(modelFileName)
del model  # deletes the existing model variable
print('model saved as', modelFileName)
```

    model saved as models/animal_classifier.h5
    

## Use the trained model

Now that we've trained your model, modify the following code as necessary to use it to predict the classes of the provided test images.


```python
# Function to predict the class of an image
from tensorflow.keras import models
import os
%matplotlib inline

def predict_image(classifier, image):
    import numpy
    from tensorflow import convert_to_tensor
    # Default value
    index = 0
    
    # !!Add your code here to predict an image class from your model!!
    imgfeatures = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])

    # We need to format the input to match the training data
    # The generator loaded the values as floating point numbers
    # and normalized the pixel values, so...
    imgfeatures = imgfeatures.astype('float32')
    imgfeatures /= 255
    
    # Use the model to predict the image class
    class_probabilities = classifier.predict(imgfeatures)
    
    # Find the class predictions with the highest predicted probability
    index = int(np.argmax(class_probabilities, axis=1)[0])
   
    # Return the predicted index
    return index


# Load your model
model = models.load_model(modelFileName)

# The images are in the data/shapes folder
test_data_path = 'data/safari/test'

# Show the test images with predictions
fig = plt.figure(figsize=(8, 12))
i = 0
for img_file in os.listdir(test_data_path):
    i+=1
    img_path = os.path.join(test_data_path, img_file)
    img = mpimg.imread(img_path)
    # Get the image class prediction
    index = predict_image(model, np.array(img))
    a=fig.add_subplot(1, len(classes),i)
    a.axis('off')
    imgplot = plt.imshow(img)
    a.set_title(classes[index])
plt.show()
```


    
![png]({{ site.baseurl }}/assets/img/2022-03-24-safari-cnn-challenge_files/2022-03-24-safari-cnn-challenge_13_0.png)
    


Hopefully, your model predicted all four of the image classes correctly!
