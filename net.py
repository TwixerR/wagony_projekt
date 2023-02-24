import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import os

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

BEST_MODEL_PATH = r"C:\Users\Grzegorz\Inzynierka\some_model"

def init():

    return

TRAIN_PATH = r"E:\2021-wagony-final (1)\wagony\train_small"

img_height=128
img_width=128
epochs=10
labels = [x[0].split("\\")[-1] for x in os.walk(TRAIN_PATH)][1:]
num_classes = len([x[0] for x in os.walk(TRAIN_PATH)]) - 1



def runall():
    # model = Sequential([
    #     layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 1)),
    #     layers.Conv2D(16, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(32, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(64, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Flatten(),
    #     layers.Dense(128, activation='relu'),
    #     layers.Dense(num_classes)
    # ])

    tiny_model = Sequential([
        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 1)),
        layers.Conv2D(8, 4, padding='same', activation='relu'),
        layers.AveragePooling2D(pool_size=(4, 4)),
        layers.Flatten(),
        layers.Dense(64),
        layers.Dense(num_classes)
    ], name="tiny_model")
    small_model = Sequential([
        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 1)),
        layers.Conv2D(8, 4, padding='same', activation='relu'),
        layers.AveragePooling2D(pool_size=(4, 4)),
        layers.Conv2D(4, 4, padding='same', activation='relu'),
        layers.AveragePooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes)
    ], name="small_model")
    average_model = Sequential([
        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 1)),
        layers.Conv2D(8, 4, padding='same', activation='relu'),
        layers.AveragePooling2D(pool_size=(4, 4)),
        layers.Conv2D(4, 4, padding='same', activation='relu'),
        layers.AveragePooling2D(pool_size=(2, 2)),
        layers.Conv2D(4, 4, padding='same', activation='relu'),
        layers.AveragePooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ], name="average_model")
    large_model = Sequential([
        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 1)),
        layers.Conv2D(8, 4, padding='same', activation='relu'),
        layers.AveragePooling2D(pool_size=(4, 4)),
        layers.Conv2D(4, 4, padding='same', activation='relu'),
        layers.AveragePooling2D(pool_size=(2, 2)),
        layers.Conv2D(4, 4, padding='same', activation='relu'),
        layers.AveragePooling2D(pool_size=(2, 2)),
        layers.Conv2D(2, 2, padding='same', activation='relu'),
        layers.AveragePooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes)
    ], name="large_model")

    models = [tiny_model, small_model, average_model, large_model]

    hists = []
    for mdl in models:
        hists.append(build_compile_train(mdl))
    return hists

def build_compile_train(model=None):
    num_classes=0

    train_ds = tf.keras.utils.image_dataset_from_directory(TRAIN_PATH,
                                                           labels='inferred',
                                                           color_mode='grayscale',
                                                           batch_size=8,
                                                           image_size=(128,128),
                                                           shuffle=True,
                                                           seed=123,
                                                           validation_split=0.2,
                                                           subset='training',
                                                           interpolation='nearest')
    validate_ds = tf.keras.utils.image_dataset_from_directory(TRAIN_PATH,
                                                           labels='inferred',
                                                           color_mode='grayscale',
                                                           batch_size=8,
                                                           image_size=(img_height,img_width),
                                                           shuffle=True,
                                                           seed=123,
                                                           validation_split=0.2,
                                                           subset='validation',
                                                           interpolation='nearest')

    num_classes = len(train_ds.class_names)
    print(f"Amount of classes loaded: {num_classes}")
    # prefetch and cache to reduce total runtime
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    validate_ds.cache().prefetch(buffer_size=AUTOTUNE)

    if model is None:
        model = Sequential([
            layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 1)),
            layers.AveragePooling2D(pool_size=(4, 4)),
            layers.Conv2D(1, kernel_size=4, strides=1, padding='same', activation='sigmoid'),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
        ])

    model.compile(optimizer='adam',
                  loss= tf.keras.losses.BinaryFocalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=validate_ds,
        epochs=epochs
    )

    return history

def predict(image, model=None):
    if model is None:
        model = tf.keras.models.load_model(BEST_MODEL_PATH)
    # case image passed as np.ndarray
    if type(image) == np.ndarray:
        img_array = image[tf.newaxis, ..., tf.newaxis]
        img_array = tf.image.resize(img_array, [img_height, img_width])
        # TODO: scale to size
    # case path to img
    elif type(image) == str:
        img = tf.keras.utils.load_img(
            image, target_size=(img_height, img_width),
            grayscale=True)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # return predicted class and certainty
    return labels[np.argmax(score)], 100 * np.max(score)
