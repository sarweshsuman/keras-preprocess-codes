from keras.utils import np_utils

def prepare_image_dataset(x,y):
    x = x.astype('float32')/255.0 # Normalizing
    y = np_utils.to_categorical(y)  # converting it from 9 to [0 0 0 0 0 0 0 0 1 0] in 10 classes
    return (x,y)
