#note to self, add optimizers in dispatcher
import json
import function_dispatcher as function_dispatcher
with open('distillation.json', 'r') as ftrain:
    data = json.load(ftrain)

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import tensorflow.keras as keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import tensorflow as tf
from keras.models import load_model
from keras.datasets import cifar100, cifar10
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.applications import EfficientNetB0, DenseNet201, EfficientNetB7, MobileNetV2, DenseNet121
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from function_dispatcher import function_dispatcher
import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def input_preprocess(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

def get_kd_loss(student_logits, teacher_logits,
                    true_labels, temperature,
                    alpha, beta):
        teacher_probs = tf.nn.softmax(teacher_logits / temperature)
        student_probs = tf.nn.softmax(student_logits / temperature)
        #print(teacher_probs)
        #print(student_logits)
        kl = tf.keras.losses.KLDivergence()
        kd_loss = tf.math.reduce_mean(tf.keras.losses.categorical_crossentropy(teacher_probs, student_probs))
        #print(kd_loss)
        kd_loss = kd_loss*(temperature**2) 
        #print(kd_loss)
        #kd_loss
        ce_loss = tf.math.reduce_mean(tf.keras.losses.categorical_crossentropy(
            true_labels, student_logits, from_logits=True))
        #print(ce_loss)
        total_loss = (alpha * kd_loss) + (beta * ce_loss)
        return total_loss

def unfreeze_model(model, layer_num=-20):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[layer_num:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

class Student(tf.keras.Model):
        def __init__(self, trained_teacher, student,
                     temperature, alpha, beta):
            super(Student, self).__init__()
            self.trained_teacher = trained_teacher
            print(trained_teacher.layers[0].get_output_at(0).get_shape().as_list())
            self.student = student
            self.temperature = temperature
            self.alpha = alpha
            self.beta = beta
            IMG_SIZE = self.trained_teacher.layers[0].get_output_at(0).get_shape().as_list()[2]
            self.size = (IMG_SIZE, IMG_SIZE)
        def train_step(self, data):
            images, labels = data
            teacher_logits = self.trained_teacher(tf.image.resize(images, self.size))
            print(images.shape)
            unfreeze_model(self.student)
            with tf.GradientTape() as tape:
                student_logits = self.student(images)
                loss = get_kd_loss(student_logits, teacher_logits,
                                    labels, self.temperature,
                                    self.alpha, self.beta)
                #loss = tf.keras.losses.categorical_crossentropy(labels, student_logits, from_logits=True)
                gradients = tape.gradient(loss, self.student.trainable_variables)
            #tf.print(loss)
            #gradients = [gradient * (self.temperature ** 2) for gradient in gradients]
            self.student.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))

            train_loss.update_state(loss)
            train_acc.update_state(labels, tf.nn.softmax(student_logits))
            t_loss, t_acc = train_loss.result(), train_acc.result()
            train_loss.reset_states(), train_acc.reset_states()
            return {"train_loss": t_loss, "train_acc": t_acc}

        def test_step(self, data):
            images,labels = data
            teacher_logits = self.trained_teacher(tf.image.resize(images, self.size))
            student_logits = self.student(images, training=False)
            loss = get_kd_loss(student_logits, teacher_logits,
                                   labels, self.temperature,
                                   self.alpha, self.beta)

            valid_loss.update_state(loss)
            valid_acc.update_state(labels, tf.nn.softmax(student_logits))
            v_loss, v_acc = valid_loss.result(), valid_acc.result()
            valid_loss.reset_states(), valid_acc.reset_states()
            return {"loss": v_loss, "accuracy": v_acc}


for i in range(1):
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

        #model = tf.keras.models.load_model("ResNet101_stl[:50%].h5")
        #student_model = tf.keras.Model(inputs, outputs)

        teacher = tf.keras.models.load_model("../training/" + data[model]["teacher"], compile=False)

        stu_model = function_dispatcher(data[model]["student_architecture"])(include_top=False, weights=data[model]["weights"], input_tensor=inputs)
        #unfreeze_model(model) 
        stu_model.trainable = False

        # Rebuild top
        x = layers.GlobalAveragePooling2D(name="avg_pool")(stu_model.output)
        x = layers.BatchNormalization()(x)
    
        top_dropout_rate = 0.2
        x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
        outputs = layers.Dense(NUM_CLASSES, activation=None, name="logit")(x)
        
        student_model = tf.keras.Model(inputs, outputs)
        unfreeze_model(student_model)
        train_loss = tf.keras.metrics.Mean(name="train_loss")
        valid_loss = tf.keras.metrics.Mean(name="test_loss")

        train_acc = tf.keras.metrics.CategoricalAccuracy(name="train_acc")
        valid_acc = tf.keras.metrics.CategoricalAccuracy(name="valid_acc")

        student_distilled = Student(teacher, student_model,temperature=data[model]["hyperparameters"]["temperature"], alpha=data[model]["hyperparameters"]["alpha"], beta=data[model]["hyperparameters"]["beta"])
        optimizer = tf.keras.optimizers.Adam(learning_rate=data[model]["hyperparameters"]["learning rate"])
        student_distilled.compile(optimizer)
        student_distilled.student.compile(optimizer)
        student_distilled.fit(ds_train,
                    validation_data= ds_test,
                    epochs=data[model]["hyperparameters"]["epoch"])
        student_distilled.student.save(data[model]["output"])

