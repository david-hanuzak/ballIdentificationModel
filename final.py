import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from sklearn import metrics

trainDir = 'archive\\train'
testDir = 'archive\\test'

input_shape = (224, 224, 3)

data = pd.DataFrame()
x = []
y = []

#iterate through input files
for ball in os.listdir(trainDir):
    for filename in os.listdir(os.path.join(trainDir,ball)):

        #convert image into tensor with dimmensions 244,244,3
        image = tf.io.read_file(os.path.join(trainDir,ball,filename))
        image = tf.io.decode_jpeg(image, channels=input_shape[-1])
        image = tf.image.resize(image, input_shape[:2])
        image = tf.cast(image, tf.float32) / 255.0

        #add converted image and classification to data
        x.append(image)
        y.append(ball)

for ball in os.listdir(testDir):
    for filename in os.listdir(os.path.join(testDir,ball)):

        # convert image into tensor with dimmensions 244,244,3
        image = tf.io.read_file(os.path.join(testDir, ball, filename))
        image = tf.io.decode_jpeg(image, channels=input_shape[-1])
        image = tf.image.resize(image, input_shape[:2])
        image = tf.cast(image, tf.float32) / 255.0

        # add converted image and classification to data
        x.append(image)
        y.append(ball)


#convert inputs into correct format for model training
x = np.asarray(x).astype('float32')
data['y'] = y
y = pd.factorize(data['y'])[0]
y = tf.one_hot(y,15)
y = np.asarray(y).astype('float32')

#split inputs into train and test data
x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True)

#define model
model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(15, activation="softmax")
])

#compile model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

#train model on training data
model.fit(x_train, y_train, epochs=10, batch_size=32)

#save model for future use
model.save('Weights_folder/Weights.h5')

#generate predictions table from testing data
predictions = model.predict(x_test)

#get predicted value from predictions table
predicted = []
for p in predictions:
    predicted.append(np.argmax(p,axis=0))

#get actual value from model input
actual = []
for yt in y_test:
    actual.append(np.argmax(yt,axis=0))

#report loss and accuracy of model when evaluated on test data
loss, accuracy = model.evaluate(x_test,y_test)
print('Testing loss:', loss)
print('Testing accuracy:', accuracy)

#create and display confusion matrix using actual and predicted values from the test data
confusion_matrix = metrics.confusion_matrix(actual, predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
cm_display.plot()
plt.show()