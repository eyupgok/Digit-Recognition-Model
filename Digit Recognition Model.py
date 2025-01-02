# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt


# %%

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


# %%

model = Sequential()
model.add(Flatten(input_shape=(28, 28))) 
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))


# %%

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# %%

model.fit(x_train, y_train, epochs=6)


# %%
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

# %%
import numpy as np
predictions = model.predict(x_test)



# %%
test=876
print(f'Tahmin: {np.argmax(predictions[test])}, Gerçek: {y_test[test]}')
plt.imshow(x_test[test], cmap='gray')
plt.title(f'Test Label: {y_test[test]}, Predict: {np.argmax(predictions[test])}')
plt.axis('off')
plt.show()


# %%
from PIL import Image

# %%
image_path = 'images.jpeg'
image = Image.open(image_path)
image = image.resize((28, 28))
image = image.convert('L')
image_array = np.array(image).astype('float32') / 255.0
image_array = image_array.reshape(1, 28, 28)
prediction = model.predict(image_array)
predicted_digit = np.argmax(prediction)
print(prediction)
print(f'Predicted Number: {predicted_digit}')

# %%



