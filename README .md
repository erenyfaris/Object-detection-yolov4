
# Object Detection with YOLO
We encounter objects every day in our life. Look around, and you’ll find multiple objects surrounding you. As a human being you can easily detect and identify each object that you see. It’s natural and doesn’t take much effort. 

For computers, however, detecting objects is a task that needs a complex solution. For a computer to “detect objects” means to process an input image (or a single frame from a video) and respond with information about objects on the image and their position. In computer vision terms, we call these two tasks classification and localization. We want the computer to say what kind of objects are presented on a given image and where exactly they’re located.



## Contents
* YOLO as a real-time object detector

* YOLO as an object detector in TensorFlow & Keras

* How to train your custom YOLO object detection model
## What is YOLO?
YOLO is an acronym for “You Only Look Once” (don’t confuse it with You Only Live Once from The Simpsons). As the name suggests, a single “look” is enough to find all objects on an image and identify them.

In machine learning terms, we can say that all objects are detected via a single algorithm run. It’s done by dividing an image into a grid and predicting bounding boxes and class probabilities for each cell in a grid. In case we’d like to employ YOLO for car detection, here’s what the grid and the predicted bounding box
It’s worth noting that YOLO’s raw output contains many bounding boxes for the same object. These boxes differ in shape and size. As you can see in the image below, some boxes are better at capturing the target object whereas others offered by an algorithm perform poorly. 
* To select the best bounding box for a given object, a Non-maximum suppression (NMS) algorithm is applied. 

## Examples of YOLO applications:
object detector :
* TensorFlow 
* Keras
TensorFlow & Keras frameworks in Machine Learning

## How to run pre-trained YOLO out-of-the-box and get results:
## Deployment

To deploy this project run

```bash
  from models import Yolov4
model = Yolov4(weight_path='yolov4.weights', 
               class_name_path='class_names/coco_classes.txt')
```


## How to train your custom YOLO object detection model:
To design an object detection model, you need to know what object types you want to detect. This should be a limited number of object types that you want to create your detector for. It’s good to have a list of object types prepared as we move to the actual model development.

Ideally, you should also have an annotated dataset that has objects of your interest. This dataset will be used to train a detector and validate it. If you don’t yet have either a dataset or annotation for it

## Dataset & annotations :      Where to get data from":
The first resource I recommend is the [“50+ Object Detection Datasets from different industry domains”]() article by Abhishek Annamraju who has collected wonderful annotated datasets for industries such as Fashion, Retail, Sports, Medicine and many more.



## How to transform data from other formats to YOLO Annotations for YOLO are in the form of txt files. Each line in a txt file fol YOLO must have the following format:
## Deployment

To deploy this project run

```bash
image1.jpg 10,15,345,284,0
image2.jpg 100,94,613,814,0 31,420,220,540,1
```
Bounding box coordinates are a clear concept, but what about the class_id number that specifies the class label? Each class_id is linked with a particular class in another txt file. For example, pre-trained YOLO comes with the coco_classes.txt file which looks like this:

```bash
 person
bicycle
car
motorbike
aeroplane
bus
...
```

## Splitting data into subsets:
```bash
  from models import Yolov4
model = Yolov4(weight_path='yolov4.weights', 
               class_name_path='class_names/coco_classes.txt')
```
from utils import read_annotation_lines
 
train_lines, val_lines = read_annotation_lines('../path2annotations/annot.txt', test_size=0.1)

## Creating data generators:
```bash
  from utils import DataGenerator
 
FOLDER_PATH = '../dataset/img'
class_name_path = '../class_names/bccd_classes.txt'
 
data_gen_train = DataGenerator(train_lines, class_name_path, FOLDER_PATH)
data_gen_val = DataGenerator(val_lines, class_name_path, FOLDER_PATH)
```
```bash
  from utils import read_annotation_lines, DataGenerator
 
train_lines, val_lines = read_annotation_lines('../path2annotations/annot.txt', test_size=0.1)
 
FOLDER_PATH = '../dataset/img'
class_name_path = '../class_names/bccd_classes.txt'
 
data_gen_train = DataGenerator(train_lines, class_name_path, FOLDER_PATH)
data_gen_val = DataGenerator(val_lines, class_name_path, FOLDER_PATH)
```

## Prerequisites:
By now you should have:

* A split for your dataset;
* Two data generators initialized;
* A txt file with the classes.

## Model object initialization:
To get ready for a training job, initialize the YOLOv4 model object. Make sure that you use None as a value for the weight_path parameter. You also should provide a path to your classes txt file at this step. Here’s the initialization code that I used in my project:

```bash
class_name_path = 'path2project_folder/model_data/scans_file.txt'
 
model = Yolov4(weight_path=None, 
               class_name_path=class_name_path)
```


## Defining callbacks:
```
   # defining pathes and callbacks
 
dir4saving = 'path2checkpoint/checkpoints'
os.makedirs(dir4saving, exist_ok = True)
 
logdir = 'path4logdir/logs'
os.makedirs(logdir, exist_ok = True)
 
name4saving = 'epoch_{epoch:02d}-val_loss-{val_loss:.4f}.hdf5'
 
filepath = os.path.join(dir4saving, name4saving)
 
rLrCallBack = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',
                                             factor = 0.1,
                                             patience = 5,
                                             verbose = 1)
 
tbCallBack = keras.callbacks.TensorBoard(log_dir = logdir,
                                         histogram_freq = 0,
                                         write_graph = False,
                                         write_images = False)
 
mcCallBack_loss = keras.callbacks.ModelCheckpoint(filepath,
                                            monitor = 'val_loss',
                                            verbose = 1,
                                            save_best_only = True,
                                            save_weights_only = False,
                                            mode = 'auto',
                                            period = 1)

esCallBack = keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                          mode = 'min',
                                          verbose = 1,
                                          patience = 10) # defining pathes and callbacks
 
dir4saving = 'path2checkpoint/checkpoints'
os.makedirs(dir4saving, exist_ok = True)
 
logdir = 'path4logdir/logs'
os.makedirs(logdir, exist_ok = True)
 
name4saving = 'epoch_{epoch:02d}-val_loss-{val_loss:.4f}.hdf5'
 
filepath = os.path.join(dir4saving, name4saving)
 
rLrCallBack = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',
                                             factor = 0.1,
                                             patience = 5,
                                             verbose = 1)
 
tbCallBack = keras.callbacks.TensorBoard(log_dir = logdir,
                                         histogram_freq = 0,
                                         write_graph = False,
                                         write_images = False)
 
mcCallBack_loss = keras.callbacks.ModelCheckpoint(filepath,
                                            monitor = 'val_loss',
                                            verbose = 1,
                                            save_best_only = True,
                                            save_weights_only = False,
                                            mode = 'auto',
                                            period = 1)

esCallBack = keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                          mode = 'min',
                                          verbose = 1,
                                          patience = 10)
                                          
```

## Fitting the model:
To kick off the training job, simply fit the model object using the standard fit() method in TensorFlow / Keras. Here’s how I started training my model:


```
 model.fit(data_gen_train, 
          initial_epoch=0,
          epochs=10000, 
          val_data_gen=data_gen_val,
          callbacks=[rLrCallBack,
                     tbCallBack,
                     mcCallBack_loss,
                     esCallBack,
                     neptune_cbk]
         ) 
```

## Model definition:
I refered to the official ResNet implementation in Tensorflow in terms of how to arange the code.

Batch norm and fixed padding¶
It's useful to define batch_norm function since the model uses batch norms with shared parameters heavily. Also, same as ResNet, Yolo uses convolution with fixed padding, which means that padding is defined only by the size of the kernel.

```
def batch_norm(inputs, training, data_format):
    """Performs a batch normalization using a standard set of parameters."""
    return tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
        scale=True, training=training)


def fixed_padding(inputs, kernel_size, data_format):
    """ResNet implementation of fixed padding.

    Pads the input along the spatial dimensions independently of input size.

    Args:
        inputs: Tensor input to be padded.
        kernel_size: The kernel to be used in the conv2d or max_pool2d.
        data_format: The input format.
    Returns:
        A tensor with the same format as the input.
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end],
                                        [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, data_format, strides=1):
    """Strided 2-D convolution with explicit padding."""
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    return tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size,
        strides=strides, padding=('SAME' if strides == 1 else 'VALID'),
        use_bias=False, data_format=data_format)
```

## Feature extraction:
```
def darknet53_residual_block(inputs, filters, training, data_format,
                             strides=1):
    """Creates a residual block for Darknet."""
    shortcut = inputs

    inputs = conv2d_fixed_padding(
        inputs, filters=filters, kernel_size=1, strides=strides,
        data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs = conv2d_fixed_padding(
        inputs, filters=2 * filters, kernel_size=3, strides=strides,
        data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs += shortcut

    return inputs


def darknet53(inputs, training, data_format):
    """Creates Darknet53 model for feature extraction."""
    inputs = conv2d_fixed_padding(inputs, filters=32, kernel_size=3,
                                  data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
    inputs = conv2d_fixed_padding(inputs, filters=64, kernel_size=3,
                                  strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs = darknet53_residual_block(inputs, filters=32, training=training,
                                      data_format=data_format)

    inputs = conv2d_fixed_padding(inputs, filters=128, kernel_size=3,
                                  strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    for _ in range(2):
        inputs = darknet53_residual_block(inputs, filters=64,
                                          training=training,
                                          data_format=data_format)

    inputs = conv2d_fixed_padding(inputs, filters=256, kernel_size=3,
                                  strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    for _ in range(8):
        inputs = darknet53_residual_block(inputs, filters=128,
                                          training=training,
                                          data_format=data_format)

    route1 = inputs

    inputs = conv2d_fixed_padding(inputs, filters=512, kernel_size=3,
                                  strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    for _ in range(8):
        inputs = darknet53_residual_block(inputs, filters=256,
                                          training=training,
                                          data_format=data_format)

    route2 = inputs

    inputs = conv2d_fixed_padding(inputs, filters=1024, kernel_size=3,
                                  strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    for _ in range(4):
        inputs = darknet53_residual_block(inputs, filters=512,
                                          training=training,
                                          data_format=data_format)
         return route1, route2, inputs
         
```

## Final model class:
```
  class Yolo_v4:
    """Yolo v4 model class."""

    def __init__(self, n_classes, model_size, max_output_size, iou_threshold,
                 confidence_threshold, data_format=None):
        """Creates the model.

        Args:
            n_classes: Number of class labels.
            model_size: The input size of the model.
            max_output_size: Max number of boxes to be selected for each class.
            iou_threshold: Threshold for the IOU.
            confidence_threshold: Threshold for the confidence score.
            data_format: The input format.

        Returns:
            None.
        """
        if not data_format:
            if tf.test.is_built_with_cuda():
                data_format = 'channels_first'
            else:
                data_format = 'channels_last'

        self.n_classes = n_classes
        self.model_size = model_size
        self.max_output_size = max_output_size
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.data_format = data_format

    def __call__(self, inputs, training):
        """Add operations to detect boxes for a batch of input images.

        Args:
            inputs: A Tensor representing a batch of input images.
            training: A boolean, whether to use in training or inference mode.

        Returns:
            A list containing class-to-boxes dictionaries
                for each sample in the batch.
        """
        with tf.variable_scope('yolo_v3_model'):
            if self.data_format == 'channels_first':
                inputs = tf.transpose(inputs, [0, 3, 1, 2])

            inputs = inputs / 255

            route1, route2, inputs = darknet53(inputs, training=training,
                                               data_format=self.data_format)

            route, inputs = yolo_convolution_block(
                inputs, filters=512, training=training,
                data_format=self.data_format)
            detect1 = yolo_layer(inputs, n_classes=self.n_classes,
                                 anchors=_ANCHORS[6:9],
                                 img_size=self.model_size,
                                 data_format=self.data_format)

            inputs = conv2d_fixed_padding(route, filters=256, kernel_size=1,
                                          data_format=self.data_format)
            inputs = batch_norm(inputs, training=training,
                                data_format=self.data_format)
            inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
            upsample_size = route2.get_shape().as_list()
            inputs = upsample(inputs, out_shape=upsample_size,
                              data_format=self.data_format)
            axis = 1 if self.data_format == 'channels_first' else 3
            inputs = tf.concat([inputs, route2], axis=axis)
            route, inputs = yolo_convolution_block(
                inputs, filters=256, training=training,
                data_format=self.data_format)
            detect2 = yolo_layer(inputs, n_classes=self.n_classes,
                                 anchors=_ANCHORS[3:6],
                                 img_size=self.model_size,
                                 data_format=self.data_format)

            inputs = conv2d_fixed_padding(route, filters=128, kernel_size=1,
                                          data_format=self.data_format)
            inputs = batch_norm(inputs, training=training,
                                data_format=self.data_format)
            inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
            upsample_size = route1.get_shape().as_list()
            inputs = upsample(inputs, out_shape=upsample_size,
                              data_format=self.data_format)
            inputs = tf.concat([inputs, route1], axis=axis)
            route, inputs = yolo_convolution_block(
                inputs, filters=128, training=training,
                data_format=self.data_format)
            detect3 = yolo_layer(inputs, n_classes=self.n_classes,
                                 anchors=_ANCHORS[0:3],
                                 img_size=self.model_size,
                                 data_format=self.data_format)

            inputs = tf.concat([detect1, detect2, detect3], axis=1)

            inputs = build_boxes(inputs)

            boxes_dicts = non_max_suppression(
                inputs, n_classes=self.n_classes,
                max_output_size=self.max_output_size,
                iou_threshold=self.iou_threshold,
                confidence_threshold=self.confidence_threshold)

            return boxes_dicts
```
