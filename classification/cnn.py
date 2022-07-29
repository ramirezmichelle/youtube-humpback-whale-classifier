from tensorflow import keras

workspace_path = '/mount/data'

class CNN:
    
    '''Class to contain various types of CNN-based Feature Extractors'''
    def __init__(self, IMG_SIZE):
        self.IMG_SIZE = IMG_SIZE
        

    def InceptionV3(self):
        '''Returns InceptionV3 architecture pre-trained on ImageNet-1k dataset for feature extraction'''

        #instantiate InceptionV3 as feature extractor 
        #we don't include the top layer since we're not using this for 
        #classification - just feature extraction)
        feature_extractor = keras.applications.InceptionV3(
                                                            weights      = 'imagenet',
                                                            include_top  = False,
                                                            pooling      = 'avg',
                                                            input_shape  = (self.IMG_SIZE, self.IMG_SIZE, 3)
                                                        )


        #required preprocessing for inceptionV3 - scales input pixels between -1 and 1
        preprocess_input = keras.applications.inception_v3.preprocess_input

        inputs = keras.Input((self.IMG_SIZE, self.IMG_SIZE, 3))
        preprocessed = preprocess_input(inputs)

        outputs = feature_extractor(preprocessed)
        return keras.Model(inputs, outputs, name="feature_extractor")
    
    def ResNet50(self):
        feature_extractor = keras.applications.ResNet50(
                                                            weights      = 'imagenet',
                                                            include_top  = False,
                                                            pooling      = 'avg',
                                                            input_shape  = (self.IMG_SIZE, self.IMG_SIZE, 3)
                                                        )


        #required preprocessing for resnet. converts images from RGB to BGR and zero-centers each color channel (w/o scaling)
        preprocess_input = keras.applications.resnet.preprocess_input

        inputs = keras.Input((self.IMG_SIZE, self.IMG_SIZE, 3))
        preprocessed = preprocess_input(inputs)

        outputs = feature_extractor(preprocessed)
        return keras.Model(inputs, outputs, name="feature_extractor")
    
    def ResNet101(self):
        feature_extractor = keras.applications.ResNet101(
                                                            weights      = 'imagenet',
                                                            include_top  = False,
                                                            pooling      = 'avg',
                                                            input_shape  = (self.IMG_SIZE, self.IMG_SIZE, 3)
                                                        )


        #required preprocessing for resnet. converts images from RGB to BGR and zero-centers each color channel (w/o scaling)
        preprocess_input = keras.applications.resnet.preprocess_input

        inputs = keras.Input((self.IMG_SIZE, self.IMG_SIZE, 3))
        preprocessed = preprocess_input(inputs)

        outputs = feature_extractor(preprocessed)
        return keras.Model(inputs, outputs, name="feature_extractor")
    
    
    def VGG16(self):
        
        feature_extractor = keras.applications.VGG16(
                                                            weights      = 'imagenet',
                                                            include_top  = False,
                                                            pooling      = 'avg',
                                                            input_shape  = (self.IMG_SIZE, self.IMG_SIZE, 3)
                                                        )


        #required preprocessing for resnet. converts images from RGB to BGR and zero-centers each color channel (w/o scaling)
        preprocess_input = keras.applications.vgg16.preprocess_input

        inputs = keras.Input((self.IMG_SIZE, self.IMG_SIZE, 3))
        preprocessed = preprocess_input(inputs)

        outputs = feature_extractor(preprocessed)
        return keras.Model(inputs, outputs, name="feature_extractor")
    
    def VGG19(self):
        
        feature_extractor = keras.applications.VGG19(
                                                            weights      = 'imagenet',
                                                            include_top  = False,
                                                            pooling      = 'avg',
                                                            input_shape  = (self.IMG_SIZE, self.IMG_SIZE, 3)
                                                        )


        #required preprocessing for resnet. converts images from RGB to BGR and zero-centers each color channel (w/o scaling)
        preprocess_input = keras.applications.vgg19.preprocess_input

        inputs = keras.Input((self.IMG_SIZE, self.IMG_SIZE, 3))
        preprocessed = preprocess_input(inputs)

        outputs = feature_extractor(preprocessed)
        return keras.Model(inputs, outputs, name="feature_extractor")


