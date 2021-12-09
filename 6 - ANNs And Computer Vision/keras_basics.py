from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from numpy import genfromtxt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# import data from .txt file
data = genfromtxt('bank_note_data.txt', delimiter=',')
y = labels = data[:, -1]
X = features = data[:, :-1]

# create train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# scale data
scaler_object = MinMaxScaler()
scaler_object.fit(X_train)
scaled_X_train = scaler_object.transform(X_train)
scaled_X_test = scaler_object.transform(X_test)

# create ANN
model = Sequential()
model.add(Dense(4, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model on the train data
model.fit(scaled_X_train, y_train, epochs=50, verbose=2)

# Run the prediction on the test set
predictions = model.predict_classes(scaled_X_test)
confusion_matrix = confusion_matrix(y_test, predictions)
classification_report = classification_report(y_test, predictions)
print(confusion_matrix)
print(classification_report)
