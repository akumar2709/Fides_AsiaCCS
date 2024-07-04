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
import csv
from tensorflow.keras.models import Sequential
import tensorflow_datasets as tfds
import random

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

service = tf.keras.models.load_model("ResNet152_cifar10_distilled.h5")

def FGSM_preprocess(images, labels):
    advs = tf.zeros([0,224,224,3])
    for image, label in zip(images, labels):    
        image = tf.reshape(image, [1,224,224,3])
        label = tf.reshape(label, [1,10])
        #prediction = tf.nn.softmax(service(image))
        #min_label = tf.one_hot(tf.math.argmin(prediction[0]), tf.size(prediction))
        #print(min_label)
        #print(tf.one_hot(tf.math.argmax(prediction[0]), tf.size(prediction)))
        signed_image = tf.identity(image)
        with tf.GradientTape() as tape:
            tape.watch(image)
            prediction = tf.nn.softmax(service(image))
            #print(tf.size(prediction))
            #print(prediction)
            #print(tf.math.argmin(prediction[0]))
            #min_label = tf.one_hot(tf.math.argmin(prediction[0]), tf.size(prediction))
            loss = tf.keras.losses.categorical_crossentropy(label, prediction)
        gradient = tape.gradient(loss,image)
        signed_image = image + 80*tf.sign(gradient)
            #print(image)
            #print(signed_grad)
        #print(image)
        #print(signed_image)
        advs = tf.concat([advs, signed_image], 0)
    return advs

# One-hot / categorical encoding
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

#ds_train_2 = ds_train_1.map(
#    FGSM_preprocess, num_parallel_calls=tf.data.AUTOTUNE
#)
#ds_train_2 = ds_train_2.batch(batch_size=batch_size, drop_remainder=True)
#ds_train_2 = ds_train_2.prefetch(tf.data.AUTOTUNE)

#ds_test_2 = ds_test_2.map(input_preprocess)
#ds_test_2 = ds_test_2.batch(batch_size=batch_size, drop_remainder=True)

train_loss = tf.keras.metrics.Mean(name="train_loss")
valid_loss = tf.keras.metrics.Mean(name="test_loss")
corrector_acc = tf.keras.metrics.CategoricalAccuracy(name='corrector_acc')
corrector_prec = tf.keras.metrics.Precision(name='corrector_prec')
corrector_rec = tf.keras.metrics.Recall(name='corrector_rec')
train_acc = tf.keras.metrics.BinaryAccuracy(name="train_acc", threshold=0.5)
valid_acc = tf.keras.metrics.BinaryAccuracy(name="valid_acc", threshold=0.5)
switch_acc = tf.keras.metrics.BinaryAccuracy(name="switch_acc", threshold=0.5)
avg_acc = tf.keras.metrics.BinaryAccuracy(name="avg_acc", threshold=0.5)
FGSM_acc = tf.keras.metrics.BinaryAccuracy(name="FGSM_acc", threshold=0.5)
backdoor_acc = tf.keras.metrics.BinaryAccuracy(name="backdoor_acc", threshold=0.5)
switch_prec = tf.keras.metrics.Precision(name='switch_prec')
switch_rec = tf.keras.metrics.Recall(name='switch_rec')
avg_prec = tf.keras.metrics.Precision(name='avg_prec')
avg_rec = tf.keras.metrics.Recall(name='avg_rec')
FGSM_prec = tf.keras.metrics.Precision(name='FGSM_prec')
FGSM_rec = tf.keras.metrics.Recall(name='FGSM_rec')
backdoor_prec = tf.keras.metrics.Precision(name='backdoor_prec')
backdoor_rec = tf.keras.metrics.Recall(name='backdoor_rec')

def GACN_model(output_size):
    model = models.Sequential()
    model.add(layers.Dense(2*output_size, activation='LeakyReLU'))
    #model.add(layers.batchNormalization())
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
    
    #predict_loss = tf.keras.losses.categorical_crossentropy(tf.nn.softmax(switch_preds), GACN_pred, from_logits=True)
    predict_loss = cross_entropy(tf.ones_like(attack_output), pred)
    loss = des_loss + predict_loss
    return loss

def dis_loss(real_output, attack_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    attack_loss = cross_entropy(tf.zeros_like(attack_output), attack_output)
    total_loss = real_loss + attack_loss
    return total_loss

def random_switch(ver_outs):
    labels = tf.zeros([0,NUM_CLASSES])
    for label in ver_outs:
        max_index = tf.argmax(label)
        classes = list(range(NUM_CLASSES))
        classes.remove(max_index.numpy())
        max_value = label[max_index]
        switch_index = tf.constant(random.choice(classes), dtype=tf.int64)
        switch_value = label[switch_index]
        tensor_without_max = tf.tensor_scatter_nd_update(label, [[max_index]], [-float('inf')])
        label = tf.tensor_scatter_nd_update(label, [[max_index], [switch_index]], [switch_value, max_value])
        labels =  tf.concat([labels, [label]], 0)
    return labels

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
            #real_output = self.discriminator(tf.concat([service_softmax, verification_softmax, loss_output], 1))
            #attack_output = self.discriminator(tf.concat([generator_softmax, verification_softmax, loss_generator], 1))
            #corrector_output = self.corrector(tf.concat([generator_softmax, verification_softmax, loss_generator], 1))
            
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
        # train_loss.update_state(gen_loss)
        # #train_acc.update_state(labels, tf.nn.softmax(student_logits))
        # t_loss = train_loss.result()
        # train_loss.update_state(detect_loss)
        # train_loss.reset_states()
        # #train_acc.update_state(labels, tf.nn.softmax(student_logits))
        # t_loss_2 = train_loss.result()
        # train_loss.reset_states()
        return {"GACN loss": gen_loss, "Det loss": detect_loss, "Det Acc":v_acc, "Corrector Loss":correction_loss}

    def test_step(self, data):
        images, labels = data
        adv = FGSM_preprocess(images, labels)
        #adv_images = FGSM(data, self.service_model)
        #corrector_labels = tf.concat([labels, labels], 0)
        verification_output = self.verification_model(images)
        service_output = self.service_model(images)
        service_adv_output = self.service_model(adv)
        verification_softmax = tf.nn.softmax(verification_output)
        service_softmax = tf.nn.softmax(service_output)
        service_adv_softmax = tf.nn.softmax(service_adv_output)
        corrector_labels = correction_label(service_softmax, verification_softmax, labels)
        corrector_labels = tf.concat([corrector_labels, corrector_labels], 0)
        loss_output = tf.reshape(tf.keras.losses.categorical_crossentropy(service_softmax, verification_softmax), [64,1])
        loss_adv = tf.reshape(tf.keras.losses.categorical_crossentropy(service_adv_softmax, verification_softmax), [64,1])
        switch_attack = switch_vals(service_softmax)
        random_attack = random_switch(service_softmax)
        random_loss = tf.reshape(tf.keras.losses.categorical_crossentropy(random_attack, verification_softmax), [64,1])
        loss_switch = tf.reshape(tf.keras.losses.categorical_crossentropy(switch_attack, verification_softmax), [64,1])

        avg_attack = avg_vals(service_softmax)
        loss_avg = tf.reshape(tf.keras.losses.categorical_crossentropy(avg_attack, verification_softmax), [64,1])
        
        real_output = self.discriminator(loss_output)
        switch_output = self.discriminator(loss_switch)
        avg_output = self.discriminator(loss_avg)
        random_output = self.discriminator(random_loss)
        FGSM_output = self.discriminator(loss_adv)
        switch_corrector_output = self.corrector(loss_switch)
        avg_corrector_output = self.corrector(loss_avg)

        detect_loss = dis_loss(labels, switch_output)

        labels = tf.ones_like(real_output)
        labels_switch = tf.concat([labels, tf.zeros_like(switch_output)], 0)
        labels_avg = tf.concat([labels, tf.zeros_like(avg_output)], 0)
        labels_FGSM = tf.concat([labels, tf.zeros_like(FGSM_output)], 0)
        labels_random = tf.concat([labels, tf.zeros_like(random_output)], 0)
        labels = tf.concat([labels, tf.zeros_like(switch_output)], 0)
        labels = tf.concat([labels, tf.zeros_like(avg_output)], 0)
        labels = tf.concat([labels, tf.zeros_like(FGSM_output)], 0)
        labels = tf.concat([labels, tf.zeros_like(random_output)], 0)
        outputs = tf.concat([real_output, switch_output, avg_output, FGSM_output, random_output],0)
        outputs_switch = tf.concat([real_output, switch_output],0)
        outputs_avg = tf.concat([real_output, avg_output],0)
        outputs_FGSM = tf.concat([real_output, FGSM_output],0)
        outputs_random = tf.concat([real_output, random_output], 0)

        corrector_outputs = tf.concat([switch_corrector_output, avg_corrector_output],0)
        valid_loss.update_state(detect_loss)
        valid_acc.update_state(labels, outputs)
        switch_acc.update_state(labels_switch, outputs_switch)
        avg_acc.update_state(labels_avg, outputs_avg)
        FGSM_acc.update_state(labels_FGSM, outputs_FGSM)
        corrector_acc.update_state(corrector_labels, corrector_outputs)
        corrector_prec.update_state(corrector_labels, corrector_outputs)
        corrector_rec.update_state(corrector_labels, corrector_outputs)
        
        switch_prec.update_state(labels_switch, outputs_switch)
        avg_prec.update_state(labels_avg, outputs_avg)
        FGSM_prec.update_state(labels_FGSM, outputs_FGSM)
        random_prec.update_state(labels_random, outputs_random)
        
        switch_rec.update_state(labels_switch, outputs_switch)
        avg_rec.update_state(labels_avg, outputs_avg)
        FGSM_rec.update_state(labels_FGSM, outputs_FGSM)
        random_rec.update_state(labels_random, outputs_random)
        random_acc.update_state(labels_random, outputs_random)
        v_loss, v_acc,  c_prec, c_acc = valid_loss.result(), valid_acc.result(),  corrector_prec.result(), corrector_acc.result()
        c_rec = corrector_rec.result()
        s_acc, a_acc, f_acc, b_acc = switch_acc.result(),  avg_acc.result(), FGSM_acc.result(), random_acc.result()
        s_prec, a_prec, f_prec, b_rec = switch_prec.result(),  avg_prec.result(), FGSM_prec.result(), random_prec.result()
        s_rec, a_rec, f_rec, b_prec = switch_rec.result(),  avg_rec.result(), FGSM_rec.result(), random_rec.result()
        f1_score = 2*(c_rec*c_prec)/(c_rec + c_prec)
        switch_f1 = 2*(s_rec*s_prec)/(s_rec + s_prec)
        avg_f1 = 2*(a_rec*a_prec)/(a_rec + a_prec)
        FGSM_f1 = 2*(f_rec*f_prec)/(f_rec + f_prec)
        random_f1 = 2*(b_rec*b_prec)/(b_rec + b_prec)
        valid_loss.reset_states(), valid_acc.reset_states(),  corrector_acc.reset_states()
        return {"Accuracy": v_acc, "Switch_acc":s_acc, "Avg_acc":a_acc, "FGSM_acc":f_acc, "Random_acc":b_acc, "Switch_f1":switch_f1, "Avg_f1":avg_f1, "FGSM_f1":FGSM_f1, "Random_f1":random_f1 }
 

GACN = GACN_model(NUM_CLASSES)
detector = tf.keras.models.load_model("ResNetCifar10_detector.h5")
corrector = tf.keras.models.load_model("ResNetCifar10_corrector.h5")

service = tf.keras.models.load_model("ResNet152.h5")
verification = tf.keras.models.load_model("ResNet50_distilled.h5")

GADN_models = GADN(detector, GACN, corrector, verification, service)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
optimizer2 = tf.keras.optimizers.Adam(learning_rate=1e-3)
GADN_models.compile(optimizer, run_eagerly=True)
GADN_models.discriminator.compile(optimizer, run_eagerly=True)
GADN_models.generator.compile(optimizer, run_eagerly=True)
GADN_models.corrector.compile(optimizer2, run_eagerly=True)

GADN_models.evaluate(ds_test_1)

