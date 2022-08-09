import tensorflow as tf
from tensorflow import keras
from keras.layers import *
from keras.models import *
from keras import backend as K

class attention(Layer):
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences

        super(attention,self).__init__()

    def build(self, input_shape):
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1), initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1), initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1))
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1))

        super(attention,self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        if self.return_sequences:
            return output
        
        return K.sum(output, axis=1)
    
def train_rnn(train_dataset, val_dataset, feature_dim):
    """ Create RNN model and run training and evaluation. """
    
    tf.keras.mixed_precision.set_global_policy('float64')
    
    features_input       = keras.Input((461, feature_dim))
    x                    = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True))(features_input)
    x                    = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True))(x)
    x                    = attention(return_sequences=False)(x)
    x                    = keras.layers.Dropout(0.2)(x)
    output               = keras.layers.Dense(2, activation="softmax")(x) #2 bc 2 class categories (0,1)
    model                = keras.Model(features_input, output)

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    my_callbacks = [keras.callbacks.EarlyStopping(monitor="val_accuracy", 
                                                  patience=5,
                                                  mode="max",
                                                  min_delta = 0.01,
                                                  restore_best_weights=True)]

    # train model
    history = model.fit(train_dataset,
                        validation_data = val_dataset,
                        epochs = 15,
                        callbacks = my_callbacks,
                        verbose= 1)

    
    return model