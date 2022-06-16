# Imports
import tensorflow 
from tensorflow import keras 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt 
from tensorflow.keras.preprocessing import image
import numpy as np


# Hyperparameter variables
EPOCHS = 5
BATCH_SIZE = 10

# Loading in the datsaset 
train_img = ImageDataGenerator(rescale=1/255.0)

train_dataset = train_img.flow_from_directory("dataset", 
                                            target_size=(400, 400), 
                                            batch_size=10)


# Getting the class indices
print(train_dataset.class_indices)
# {'jump': 0, 'normal': 1}

class_names = ["jump", "normal"]


# Creating the model 
''' model = keras.Sequential([
    keras.layers.Conv2D(32, 3, activation='relu'), 
    keras.layers.MaxPooling2D(2, 2), 
    keras.layers.Flatten(), 
    keras.layers.Dense(2, activation='softmax')
]) ''' 


# Loading in the model 
model = keras.models.load_model("flappy-bird-5-epoch-v1.0/")

# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the model 
# history = model.fit(train_dataset, epochs=EPOCHS, batch_size=BATCH_SIZE)

# Getting the model summary 
model.summary()

# Saving the model 
# model.save("flappy-bird-5-epoch-v1.0/")

# Plot the loss and the accuracy over the epochs 
''' plt.plot(history.history['accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig("Accuracy Over Epochs")
plt.show()


plt.plot(history.history['loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig("Loss Over Epochs")
plt.show() '''



# Prediction Code
test_img = "prediction_image.png"
img = image.load_img(test_img, target_size=(400, 400))

X = image.img_to_array(img)
X = np.expand_dims(X, axis=[0])
images = np.vstack([X])

prediction = model.predict(images)

plt.imshow(img)
plt.title(class_names[np.argmax(prediction)])
plt.show()

print(class_names[np.argmax(prediction)])
