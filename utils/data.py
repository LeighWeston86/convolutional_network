from keras.datasets import mnist
from keras.utils import to_categorical

def get_data():
    '''
    Load and format the mnist dataset
    :return: Train and test sets for image classification
    '''

    #Load and reshape the data (single channel for grayscale)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32')
    X_test  = X_test.reshape(-1, 28, 28, 1).astype('float32')

    #Normalize
    X_train /= 255
    X_test /= 255

    #Convert labels to categorical
    y_train = to_categorical(y_train, num_classes = 10)
    y_test  = to_categorical(y_test, num_classes = 10)

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data()


