from sklearn.metrics import classification_report
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# import data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
single_image = x_train[0]

# encode numbers as categorical
y_cat_test = to_categorical(y_test, 10)
y_cat_train = to_categorical(y_train, 10)

# normalize sets
x_train = x_train/x_train.max()
x_test = x_test/x_test.max()

# reshape sets to include color channels - as with a generic CNN
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# create the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(4, 4), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten()) # from 2D to 1D
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()
model.fit(x_train, y_cat_train, epochs=2)

model.evaluate(x_test, y_cat_test)

predictions = model.predict_classes(x_test)
print(classification_report(y_test, predictions))
