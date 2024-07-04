import tensorflow as tf
import tensorflow.keras.applications as applications
models = {"ResNet50":applications.ResNet50, 
          "ResNet101":applications.ResNet101,
          "ResNet152":applications.ResNet152,
          "VGG16":applications.VGG16,
          "VGG19":applications.VGG19,
          "EfficientNetB3":applications.EfficientNetB3,
          "EfficientNetB0":applications.EfficientNetB0,
          "DenseNet121":applications.DenseNet121,
          "DenseNet201":applications.DenseNet201
          }

def function_dispatcher(name):
 return models[name]
