import tensorflow as tf
from tensorflow import keras
from cnn import FeatureExtractor
import numpy as np
import time

def get_feature_extractor(cnn_model, augment_data, frame_dim=(224, 224)): #fix to allow frame_dim tuple (H, W)
    """Returns keras CNN architecture to use as feature extractor"""

    base_models = FeatureExtractor(frame_dim[0], frame_dim[1], augment_data)

    if cnn_model == "vgg16":
        return base_models.VGG16()
    
    elif cnn_model == "vgg19":
        return base_models.VGG19()
    
    elif cnn_model == "resnet50":
        return base_models.ResNet50()
    
    elif cnn_model == "resnet101":
        return base_models.ResNet101()

    elif cnn_model == "inception":
        return base_models.InceptionV3()


def select_gpus(num_gpus):
    """ Create a list of GPU devices to use with TF Strategy """
    return [f'/GPU:{i}' for i in range(num_gpus)]


def feature_extraction_cpu(dataset, frames_per_video, cnn_choice, augment_data=False):
    """Uses a CNN to extract feature representations from video frames dataset."""

    with tf.device("/device:CPU:0"):
        cnn_model = get_feature_extractor(cnn_choice, augment_data)

        #get the size of features outputted at last layer of cnn
        num_videos = dataset.__len__().numpy()
        feature_dim = cnn_model.layers[-1].output_shape[1] 

        #init empty array to store features
        features = np.empty((num_videos, frames_per_video, feature_dim), dtype=np.uint8)
        labels = np.empty(num_videos, dtype = np.uint8)

        start = time.time()
        for i, elements in enumerate(dataset.as_numpy_iterator()):
            frame_batch, label = elements
            labels[i] = label
            features[i, ...] = cnn_model.predict_on_batch(frame_batch)

        stop = time.time()
        duration = stop - start
        print(f'Done getting video frame feature representations in {stop-start} seconds (CPU mode).')
    
    return features, labels, duration


def feature_extraction_gpu(num_gpus, dataset, cnn_choice, augment_data=False):
    
    #check for correct amoung of GPUs
    if num_gpus < 1:
        print(f"{num_gpus} GPUs is not enough to perform feature extraction in GPU mode. \
        Consider switching to CPU.")
        return
    
    # create TF strategy with selected num_gpus 
    print(f'Setting TensorFlow Mirrored Strategy with {num_gpus} GPUs...')
    active_gpus = select_gpus(num_gpus)
    strategy = tf.distribute.MirroredStrategy(active_gpus)
    print ('Number of devices in strategy: {}'.format(strategy.num_replicas_in_sync))
    
    # keep batch size at 1 to avoid memory alloc. issues since tensors are large
    BATCH_SIZE_PER_REPLICA = 1
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    
    # store data in TF dataset with batch + prefetch
    print("Creating TF Dataset...")
    with tf.device("/device:CPU:0"):
        dataset = dataset.batch(GLOBAL_BATCH_SIZE)
        dataset = dataset.prefetch(2) #* strategy.num_replicas_in_sync)
    
    # creates a distributed dataset aligned with our TF strategy
    print("Creating TF DISTRIBUTED Dataset...")
    dist_dataset = strategy.experimental_distribute_dataset(dataset)
    
    # create a feature extractor on each active GPU in our TF strategy
    print("Creating CNN Model on Each Active Replica (GPU)")
    with strategy.scope():
        cnn_model = get_feature_extractor(cnn_choice, augment_data)
        
    @tf.function
    def distributed_test_step(dataset_inputs):
        """ Distributes data inputs and invokes feature extraction on each GPU """
        return strategy.run(test_step, args=(dataset_inputs,))
    
    def test_step(inputs):
        """ Returns feature representations on inputs """
        images, label = inputs

        #in the case that data does not divide evenly across GPUs,
        #TF creates a placeholder tensor so all GPUs have something to process
        #however this tensor will be empty and have shape (0, 461, 224, 224, 3)
        #so instead of running prediction, we return None values and filter these out later
        try:
            predictions = cnn_model(tf.squeeze(images), training=False)
            return predictions, label
        except ValueError:
            return None, None
        
    print("Beginning Feature Extraction in GPU Mode...")
    start = time.time()
    
    distributed_features = []
    distributed_labels = []
    for batch in dist_dataset:
        batch_features, batch_labels = distributed_test_step(batch)
        distributed_features.append(batch_features)
        distributed_labels.append(batch_labels)

    stop = time.time()
    duration = stop - start
    print(f'Done getting video frame feature representations in {stop-start} seconds.')
    
    print("Formatting Results into Numpy Arrays...")
    features = replica_objects_to_numpy(distributed_features, num_gpus)
    labels = replica_objects_to_numpy(distributed_labels, num_gpus)
    
    return features, labels, duration

def replica_objects_to_numpy(replica_results, num_gpus):
    """ Converts a list of TF Replica Objects into a Numpy ND array """
    
    if num_gpus > 1:
        # turn PerReplica objects from multi gpu's into list of tensors
        tensors = []
        for replica_obj in replica_results:
            tensors += list(replica_obj.values)

        #convert list of tensors to list of numpy arrays
        results = []
        for tensor in tensors:
            #filter out empty tensors generated during data distribution
            if tensor is not None: 
                results.append(tensor.numpy())
            
    else:
        #convert list of tensors from single gpu to list of np arrays
        results = []
        for tensor in replica_results:
            results.append(tensor.numpy())
    
    return np.array(results)
