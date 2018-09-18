import numpy as np
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Activation, Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam
from utils.data import get_data

class ConvNet:
    def __init__(self):
        self.model = None

    def keras_model(self, input_shape, num_classes):

        #Define the input layer
        X_input = Input(input_shape)

        #1st Conv layer
        X = Conv2D(filters = 8, kernel_size = 2, strides = (1, 1), padding = 'same')(X_input)
        X = Activation('relu')(X)
        X = BatchNormalization(axis = -1)(X)
        X = MaxPooling2D((2, 2))(X)

        #2nd Conv layer
        X = Conv2D(filters = 16, kernel_size= 2, strides = (1, 1), padding = 'same')(X)
        X = Activation('relu')(X)
        X = BatchNormalization(axis = -1)(X)
        X = MaxPooling2D((2, 2))(X)

        #Flatten
        X = Flatten()(X)

        #Fully connected layers
        X = Dense(100, activation = 'relu')(X)
        X = Dense(50, activation='relu')(X)

        #Output layer
        X = Dense(num_classes, activation = 'softmax')(X)

        #Define the model
        model = Model(inputs = X_input, outputs = X)

        return model

    def fit(self, X_train, y_train, learning_rate = 0.001, epochs = 10, batch_size = 128):
        #Get the model
        self.model = self.keras_model(X_train[0].shape,y_train[0].shape[0])

        #Compile
        adam = Adam(lr = learning_rate)
        self.model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])

        #Fit the model
        self.model.fit(X_train, y_train, epochs=epochs, batch_size = batch_size)

    def predict(self, X_test):
        return self.model.predict(X_test)


if __name__ == '__main__':
    from sklearn.metrics import f1_score
    #Get the data
    X_train, X_test, y_train, y_test = get_data()

    #Fit a model
    model = ConvNet()
    model.fit(X_train, y_train, epochs = 10)

    #Assess the accuracy
    predicted = model.predict(X_test).argmax(axis = 1)
    actual = y_test.argmax(axis = 1)
    print(actual)
    print(predicted)
    print(f1_score(predicted, actual, average=None))










