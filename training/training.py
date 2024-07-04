import json

with open('training.json', 'r') as ftrain:
    data = json.load (ftrain)

from tensorflow.keras.applications import ResNet50, EfficientNetB0, DenseNet201, ResNet101, EfficientNetB3, VGG16
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from function_dispatcher import function_dispatcher

def input_preprocess(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

def unfreeze_model(model):
    for layer in model.layers:
        layer.trainable = True

for model in data:
    batch_size = data[model]["batch_size"]

    dataset_name = data[model]["dataset"]

    (ds_train, ds_test), ds_info = tfds.load(
            dataset_name, split=[data[model]["split"]["train"],data[model]["split"]["test"]], with_info=True, as_supervised=True
    )
    NUM_CLASSES = ds_info.features["label"].num_classes

    IMG_SIZE = data[model]["image_size"]

    size = (IMG_SIZE, IMG_SIZE)
    ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, size), label))
    ds_test = ds_test.map(lambda image, label: (tf.image.resize(image, size), label))
    
    ds_train = ds_train.map(
        input_preprocess, num_parallel_calls=tf.data.AUTOTUNE
    )
    ds_train = ds_train.batch(batch_size=batch_size, drop_remainder=True)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(input_preprocess)
    ds_test = ds_test.batch(batch_size=batch_size, drop_remainder=True)

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    arch = function_dispatcher(data[model]["architecture"])(include_top=True, weights=None, classifier_activation=None, classes=NUM_CLASSES)

#model = tf.keras.models.load_model("ResNet101_stl[:50%].h5")
    #arch = tf.keras.Model(inputs, outputs)
    unfreeze_model(arch)
    arch.compile(
        optimizer=data[model]["optimizer"], loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=data[model]["metrics"]
        )
    #arch.compile(
    #    optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics="accuracy"
    #)
    arch.summary()

    #epochs = data[model]["hyperparameters"]["epoch"]  # @param {type: "slider", min:10, max:100}
    epochs = data[model]["hyperparameters"]["epoch"]
    hist = arch.fit(ds_train, epochs=epochs, validation_data=ds_test, verbose=1)

    arch.save(data[model]["output"][0])
