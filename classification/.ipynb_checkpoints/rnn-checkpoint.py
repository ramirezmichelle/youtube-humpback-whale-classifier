
class RNN:
    def __init__():
        self.model= None
        
    
    def model(self, MAX_SEQ_LENGTH, NUM_FEATURES ):
    
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

    def fit(self, frame_features, frame_masks, frame_labels, model_name):
        """Fit model with frame features, masks, and label data. Weights are saved in models directory"""
        
        # only saves best weights (when best val_loss is achieved)
        filepath        = '/models/' + model_name
        my_callbacks    = [keras.callbacks.ModelCheckpoint(filepath, save_weights_only=True, save_best_only=True, verbose=0)]

        # Fit data to model  
        history = self.model.fit([frame_features, frame_masks], 
                                frame_labels,
                                validation_split = 0.1,
                                callbacks = my_callbacks,
                                epochs = 10,
                                verbose= 0
                            )
        
        return self.model

    def predict(self, frame_features, frame_masks):
        """ Returns the predicted labels on input frame features and masks
        """
        
        #generate class probabilities for each example
        probabilities = self.model.predict(frame_features, frame_masks)
        
        #take the max probability and assign the corresponding class to each example
        max_probabilities = np.array([np.argmax(p) for p in probabilities])
        
        return max_probabilities
        
    
    def evauluate(self, frame_features, frame_masks, frame_labels):
        """ Generate loss and accuracy metrics on rnn's performance 
            for the input frame features, masks, and labels
        """
        
        # Generate metrics on input frame data
        loss, accuracy = self.model.evaluate([frame_features, frame_masks], frame_labels)
        
        return loss, accuracy
        

