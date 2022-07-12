from tensorflow import keras
import numpy as np

class RNN:
    def __init__(self):
        return
        
    def build_model(self, MAX_SEQ_LENGTH, NUM_FEATURES):
    
        # create + compile the model
        class_vocab          = [0, 1] 
        frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
        mask_input           = keras.Input((MAX_SEQ_LENGTH, ), dtype="bool")

        x                    = keras.layers.GRU(16, return_sequences=True)(frame_features_input, mask=mask_input)
        x                    = keras.layers.GRU(8)(x)
        x                    = keras.layers.Dropout(0.4)(x)
        x                    = keras.layers.Dense(8, activation="relu")(x)
        output               = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

        self.model          = keras.Model([frame_features_input, mask_input], output)
        
        return
    
    def compile_model(self, loss, optimizer, metrics):
        
        #compile model
        self.model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])
        
        return

    def fit(self, frame_features, frame_masks, frame_labels, num_epochs=10, verbose=0):
        """Fit model with frame features, masks, and label data. Weights are saved in models directory"""
        
        # stop training early if accuracy is not increasing
        my_callbacks    = [keras.callbacks.EarlyStopping(monitor="val_accuracy", 
                                                         patience=5,
                                                         mode="max",
                                                         min_delta = 0.01,
                                                         restore_best_weights=True)]

        # Fit data to model  
        history = self.model.fit([frame_features, frame_masks], 
                                frame_labels,
                                validation_split = 0.2,
                                callbacks = my_callbacks,
                                epochs = num_epochs,
                                verbose= verbose
                            )
        
        return history

    def predict(self, frame_features, frame_masks):
        """ Returns the predicted labels on input frame features and masks
        """
        
        #generate class probabilities for each example
        probabilities = self.model.predict([frame_features, frame_masks])
        
        #take the max probability and assign the corresponding class to each example
        max_probabilities = np.array([np.argmax(p) for p in probabilities])
        
        return max_probabilities
        
    
    def evaluate(self, frame_features, frame_masks, frame_labels):
        """ Generate loss and accuracy metrics on rnn's performance 
            for the input frame features, masks, and labels
        """
        
        # Generate metrics on input frame data
        loss, accuracy = self.model.evaluate([frame_features, frame_masks], frame_labels)
        
        return loss, accuracy
        

