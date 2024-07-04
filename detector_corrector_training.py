import tensorflow.keras as keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import tensorflow as tf
from keras.utils import np_utils
from keras.models import load_model
from keras.datasets import cifar100, cifar10
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import csv
from tensorflow.keras.models import Sequential
import tensorflow_datasets as tfds
import random
import tensorflow_addons as tfa

cross_entropy = tf.keras.losses.BinaryCrossentropy()


img_augmentation = Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)


def input_preprocess(image, label):
    label = tf.convert_to_tensor(label, dtype=tf.int64)
    label = tf.one_hot(label, NUM_CLASSES)
    #adv_img = FGSM_preprocess(image, label)
    #print(service(tf.reshape(adv_img, [1,224,224,3])))
    #print(label)
    return image,  label

batch_size = 64

dataset_name = "Cifar10"
(ds_train_1, ds_test_1), ds_info = tfds.load(
        dataset_name, split=["train[:10%] + test[:50%]", "train[25%:35%] + test[50%:]"], with_info=True, as_supervised=True
)
NUM_CLASSES = ds_info.features["label"].num_classes


IMG_SIZE = 224

size = (IMG_SIZE, IMG_SIZE)
ds_train_1 = ds_train_1.map(lambda image, label: (tf.image.resize(image, size), label))
ds_test_1 = ds_test_1.map(lambda image, label: (tf.image.resize(image, size), label))


ds_train_1 = ds_train_1.map(
    input_preprocess, num_parallel_calls=tf.data.AUTOTUNE
)
ds_train_1 = ds_train_1.batch(batch_size=batch_size, drop_remainder=True)
ds_train_1 = ds_train_1.prefetch(tf.data.AUTOTUNE)

ds_test_1 = ds_test_1.map(
    input_preprocess, num_parallel_calls=tf.data.AUTOTUNE
)
ds_test_1 = ds_test_1.batch(batch_size=batch_size, drop_remainder=True)
ds_test_1 = ds_test_1.prefetch(tf.data.AUTOTUNE)

valid_loss = tf.keras.metrics.Mean(name="test_loss")
corrector_acc = tf.keras.metrics.CategoricalAccuracy(name='corrector_acc')
corrector_F1 = tfa.metrics.F1Score(num_classes=NUM_CLASSES, name='corrector_F1')
corrector_prec = tf.keras.metrics.Precision(name='corrector_prec')
corrector_rec = tf.keras.metrics.Recall(name='corrector_rec')
train_acc = tf.keras.metrics.BinaryAccuracy(name="train_acc", threshold=0.5)
valid_acc = tf.keras.metrics.BinaryAccuracy(name="valid_acc", threshold=0.5)
switch_acc = tf.keras.metrics.BinaryAccuracy(name="switch_acc", threshold=0.5)
avg_acc = tf.keras.metrics.BinaryAccuracy(name="avg_acc", threshold=0.5)
FGSM_acc = tf.keras.metrics.BinaryAccuracy(name="FGSM_acc", threshold=0.5)
valid_F1 = tfa.metrics.F1Score(num_classes=1, name="valid_F1", threshold=0.5)
switch_F1 = tfa.metrics.F1Score(num_classes=1, name="switch_F1", threshold=0.5)
avg_F1 = tfa.metrics.F1Score(num_classes=1, name="avg_F1", threshold=0.5)
FGSM_F1 = tfa.metrics.F1Score(num_classes=1, name="FGSM_F1", threshold=0.5)

def GACN_model(output_size):
    model = models.Sequential()
    model.add(layers.Dense(2*output_size, activation='LeakyReLU'))
    model.add(layers.Dense(64, activation='LeakyReLU'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(128, activation='LeakyReLU'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation='LeakyReLU'))
    model.add(layers.Dense(output_size))

    return model

def detection_model(output_size):
    detection_model = models.Sequential()
    detection_model.add(keras.Input(shape=(1)))
    detection_model.add(layers.Dense(128, activation='LeakyReLU'))
    detection_model.add(layers.Dropout(0.2))
    detection_model.add(layers.BatchNormalization())
    detection_model.add(layers.Dense(128, activation='LeakyReLU'))
    detection_model.add(layers.Dropout(0.2))
    detection_model.add(layers.BatchNormalization())
    detection_model.add(layers.Dense(64, activation='LeakyReLU'))
    detection_model.add(layers.Dropout(0.2))
    detection_model.add(layers.BatchNormalization())
    detection_model.add(layers.Dense(1, activation="sigmoid"))
    return detection_model

def correction_model(output_size):
    correction_model = models.Sequential()
    correction_model.add(keras.Input(shape=(1)))
    correction_model.add(layers.Dense(128, activation='LeakyReLU'))
    correction_model.add(layers.Dropout(0.2))
    correction_model.add(layers.BatchNormalization())
    correction_model.add(layers.Dense(128, activation='LeakyReLU'))
    correction_model.add(layers.Dropout(0.2))
    correction_model.add(layers.BatchNormalization())
    correction_model.add(layers.Dense(64, activation='LeakyReLU'))
    correction_model.add(layers.Dropout(0.2))
    correction_model.add(layers.BatchNormalization())
    correction_model.add(layers.Dense(5, activation="softmax"))
    return correction_model

def prediction_array(service, verification, dataset):
    service_model = tf.keras.models.load_save(service)
    verification_model = tf.keras.models.load_save(verification)
    verification_output = []
    service_output = []
    for images, labels in dataset:
        service_output = service_model(images)
        verification_output = verification_model(images)
    return [service_output, verification_output]

def correction_label(service, verification, labels):
    correction_labels = tf.zeros([0, 5], dtype=tf.int32)
    for service_label, verification_label, label in zip(service, verification, labels):
        service_max = tf.argmax(service_label)
        verification_max = tf.argmax(verification_label)
        label_max = tf.argmax(label)
        if(service_max == verification_max == label_max):
            correction_labels = tf.concat([correction_labels, [tf.constant([1, 0, 0, 0, 0], dtype=tf.int32)]], 0)
        elif(verification_max != service_max == label_max):
            correction_labels = tf.concat([correction_labels, [tf.constant([0, 1, 0, 0, 0], dtype=tf.int32)]], 0)
        elif(service_max == verification_max == label_max):
            correction_labels = tf.concat([correction_labels, [tf.constant([0, 0, 1, 0, 0], dtype=tf.int32)]], 0)
        elif(service_max != verification_max != label_max):
            correction_labels = tf.concat([correction_labels, [tf.constant([0, 0, 0, 1, 0], dtype=tf.int32)]], 0)
        else:
            correction_labels = tf.concat([correction_labels, [tf.constant([0, 0, 0, 0, 1], dtype=tf.int32)]], 0)
    return correction_labels

def corrector_loss(corrector_preds, labels):
    return tf.keras.losses.categorical_crossentropy(labels, corrector_preds)
def switch_vals(ver_outs):
    labels = tf.zeros([0,NUM_CLASSES])
    for label in ver_outs:
        max_index = tf.argmax(label)
        tensor_without_max = tf.tensor_scatter_nd_update(label, [[max_index]], [-float('inf')])
        second_max_index = tf.argmax(tensor_without_max)
        tensor_without_max = tf.tensor_scatter_nd_update(tensor_without_max, [[max_index]], [label[second_max_index]])
        label = tf.tensor_scatter_nd_update(tensor_without_max, [[second_max_index]], [label[max_index]])
        labels =  tf.concat([labels, [label]], 0)
    return labels
        
def GACN_loss(service_pred, GACN_pred, attack_output, labels):
    des_loss = cross_entropy(tf.ones_like(attack_output), attack_output)
    switch_preds = switch_vals(service_pred)
    pred = tf.zeros([0,1])
    for service, GACN in zip(service_pred, GACN_pred):
        service_max = tf.argmax(service).numpy()
        GACN_max = tf.argmax(GACN).numpy()
        GACN = tf.nn.softmax(GACN)
        if(service_max != GACN_max):
            GACN = GACN.numpy()
            GACN = GACN[GACN_max]
            pred = tf.concat([pred, [tf.constant([GACN], dtype=tf.float32)]], 0)
        else:
            service = tf.nn.softmax(service)
            service = service.numpy()
            service = 1 - service[service_max]
            pred = tf.concat([pred, [tf.constant([service], dtype=tf.float32)]], 0)
    
    predict_loss = cross_entropy(tf.ones_like(attack_output), pred)
    loss = des_loss + predict_loss
    return loss

def dis_loss(real_output, attack_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    attack_loss = cross_entropy(tf.zeros_like(attack_output), attack_output)
    total_loss = real_loss + attack_loss
    return total_loss

def avg_vals(ver_outs):
    labels = tf.zeros([0,NUM_CLASSES])
    for label in ver_outs:
        max_index = tf.argmax(label)
        tensor_without_max = tf.tensor_scatter_nd_update(label, [[max_index]], [-float('inf')])
        second_max_index = tf.argmax(tensor_without_max)
        avg_val = (label[second_max_index] + label[max_index])/2
        label = tf.tensor_scatter_nd_update(tensor_without_max, [[max_index]], [avg_val - 0.01])
        label = tf.tensor_scatter_nd_update(label, [[second_max_index]], [avg_val + 0.01])
        labels =  tf.concat([labels, [label]], 0)
    return labels

class GADN(tf.keras.Model):
    def __init__(self, detector, generator, corrector, verification_model, service_model):
        super(GADN, self).__init__()
        self.discriminator = detector
        self.generator = generator
        self.corrector = corrector
        self.verification_model = verification_model
        self.service_model = service_model

    def train_step(self, data):
        images, labels = data
        verification_output = self.verification_model(images)
        service_output = self.service_model(images)
        verification_softmax = tf.nn.softmax(verification_output)
        service_softmax = tf.nn.softmax(service_output)
        loss_output = tf.reshape(tf.keras.losses.categorical_crossentropy(service_softmax, verification_softmax), [64,1])
        corrector_labels = correction_label(service_softmax, verification_softmax, labels)
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as corr_tape:
            generator_output = self.generator(service_output)
            generator_softmax = tf.nn.softmax(generator_output)
            loss_generator = tf.reshape(tf.keras.losses.categorical_crossentropy(generator_softmax, verification_softmax), [64,1])
            
            real_output = self.discriminator(loss_output)
            attack_output = self.discriminator(loss_generator) 
            corrector_output = self.corrector(loss_generator)

            gen_loss = GACN_loss(service_output, generator_output, attack_output, labels)
            detect_loss = dis_loss(real_output, attack_output)
            correction_loss = corrector_loss(corrector_output, corrector_labels)

        gradient_gen = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradient_disc = disc_tape.gradient(detect_loss, self.discriminator.trainable_variables)
        gradient_corr = corr_tape.gradient(correction_loss, self.corrector.trainable_variables)

        self.corrector.optimizer.apply_gradients(zip(gradient_corr, self.corrector.trainable_variables))
        self.generator.optimizer.apply_gradients(zip(gradient_gen, self.generator.trainable_variables))
        self.discriminator.optimizer.apply_gradients(zip(gradient_disc, self.discriminator.trainable_variables))

        labels = tf.ones_like(real_output)
        labels = tf.concat([labels, tf.zeros_like(attack_output)], 0)
        outputs = tf.concat([real_output, attack_output],0)
        valid_acc.update_state(labels, outputs)
        v_acc  =  valid_acc.result()
        valid_acc.reset_states()
        return {"GACN loss": gen_loss, "Det loss": detect_loss, "Det Acc":v_acc, "Corrector Loss":correction_loss}

    def test_step(self, data):
        images, labels = data
        verification_output = self.verification_model(images)
        service_output = self.service_model(images)
        verification_softmax = tf.nn.softmax(verification_output)
        service_softmax = tf.nn.softmax(service_output)
        corrector_labels = correction_label(service_softmax, verification_softmax, labels)
        corrector_labels = tf.concat([corrector_labels, corrector_labels], 0)
        loss_output = tf.reshape(tf.keras.losses.categorical_crossentropy(service_softmax, verification_softmax), [64,1])
        switch_attack = switch_vals(service_softmax)
        loss_switch = tf.reshape(tf.keras.losses.categorical_crossentropy(switch_attack, verification_softmax), [64,1])

        avg_attack = avg_vals(service_softmax)
        loss_avg = tf.reshape(tf.keras.losses.categorical_crossentropy(avg_attack, verification_softmax), [64,1])
        
        real_output = self.discriminator(loss_output)
        switch_output = self.discriminator(loss_switch)
        avg_output = self.discriminator(loss_avg)
        switch_corrector_output = self.corrector(loss_switch)
        avg_corrector_output = self.corrector(loss_avg)

        detect_loss = dis_loss(labels, switch_output)

        labels = tf.ones_like(real_output)
        labels_switch = tf.concat([labels, tf.zeros_like(switch_output)], 0)
        labels_avg = tf.concat([labels, tf.zeros_like(avg_output)], 0)
        labels = tf.concat([labels, tf.zeros_like(switch_output)], 0)
        labels = tf.concat([labels, tf.zeros_like(avg_output)], 0)
        outputs = tf.concat([real_output, switch_output, avg_output],0)
        outputs_switch = tf.concat([real_output, switch_output],0)
        outputs_avg = tf.concat([real_output, avg_output],0)
        corrector_outputs = tf.concat([switch_corrector_output, avg_corrector_output],0)
        valid_loss.update_state(detect_loss)
        valid_acc.update_state(labels, outputs)
        switch_acc.update_state(labels_switch, outputs_switch)
        avg_acc.update_state(labels_avg, outputs_avg)
        valid_F1.update_state(labels, outputs)
        switch_F1.update_state(labels_switch, outputs_switch)
        avg_F1.update_state(labels_avg, outputs_avg)
        corrector_acc.update_state(corrector_labels, corrector_outputs)
        corrector_prec.update_state(corrector_labels, corrector_outputs)
        corrector_rec.update_state(corrector_labels, corrector_outputs)

        v_loss, v_acc, v_f1, c_prec, c_acc = valid_loss.result(), valid_acc.result(), valid_F1.result(), corrector_prec.result(), corrector_acc.result()
        c_rec = corrector_rec.result()
        s_acc, s_f1, a_acc, a_f1= switch_acc.result(), switch_F1.result(), avg_acc.result(), avg_F1.result()
        f1_score = 2*(c_rec*c_prec)/(c_rec + c_prec)
        valid_loss.reset_states(), valid_acc.reset_states(), corrector_F1.reset_states(), corrector_acc.reset_states()
        return {"Corrector acc":c_acc, "Corrector recall": c_rec, "Corrector Prec": c_prec, "Corrector-F1": f1_score, "Accuracy": v_acc, "F1Score":v_f1, "Switch_acc":s_acc, "Switch_F1":s_f1, "Avg_acc":a_acc, "avg-F1":a_f1} #, "FGSM_acc":f_acc, "FGSM-F1":f_f1}
 

GACN = GACN_model(NUM_CLASSES)
detector = detection_model(NUM_CLASSES)
corrector = correction_model(NUM_CLASSES)

service = tf.keras.models.load_model("training/ResNet152.h5")
verification = tf.keras.models.load_model("distillation/ResNet50_distilled.h5")

GADN_models = GADN(detector, GACN, corrector, verification, service)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
optimizer2 = tf.keras.optimizers.Adam(learning_rate=1e-3)
GADN_models.compile(optimizer, run_eagerly=True)
GADN_models.discriminator.compile(optimizer, run_eagerly=True)
GADN_models.generator.compile(optimizer, run_eagerly=True)
GADN_models.corrector.compile(optimizer2, run_eagerly=True)

GADN_models.fit(ds_train_1,validation_data= ds_test_1, epochs=1)
GADN_models.evaluate(ds_test_1)
GADN_models.generator.save("ResNetCifar10GAN_2.h5")
GADN_models.discriminator.save("ResNetCifar10_detector.h5")
GADN_models.corrector.save("ResNetCifar10_corrector.h5")


