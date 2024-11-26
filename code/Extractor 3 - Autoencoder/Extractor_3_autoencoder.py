# CAMBIAR LOS SIGUIENTES DIRECTORIOS DEPENDIENDO DE LA UBICACIÓN DE LOS DATOS

target_directory = "directorio de destino de los datos preprocesados"


from tensorflow.keras.preprocessing import image as imageprep
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import matplotlib.pyplot as plt


def image_to_array(img_path, image_size):
    """
    Convierte una imagen en una matriz numpy.

    Args:
        img_path (str): Ruta de la imagen.
        image_size (int): Tamaño al que se redimensionará la imagen.

    Returns:
        numpy.ndarray: Imagen convertida en una matriz numpy.
    """
    # Cargar la imagen y redimensionarla al tamaño especificado
    img = imageprep.load_img(img_path, target_size=(image_size, image_size))
    # Convertir la imagen a una matriz numpy
    img = imageprep.img_to_array(img)

    return img

def load_images(image_dir, image_size):
    """
    Carga todas las imágenes de un directorio y las convierte en matrices numpy.

    Args:
        image_dir (str): Directorio donde se encuentran las imágenes.
        image_size (int): Tamaño al que se redimensionarán las imágenes.

    Returns:
        list: Lista de imágenes convertidas en matrices numpy.
    """
    # Obtener la lista de archivos en el directorio de imágenes
    image_list = os.listdir(image_dir)
    img_holder = []

    for img in image_list:
        # Ignorar archivos que no sean imágenes JPG
        if '.jpg' not in img:
            continue

        # Construir la ruta completa de la imagen
        img_path = image_dir + '/' + img
        # Convertir la imagen a una matriz numpy
        img = image_to_array(img_path, image_size)
        # Añadir la imagen convertida a la lista
        img_holder.append(img)

    return img_holder


train_dir = target_directory + '/train'

# Cargar cada imagen, asegurar la compatibilidad del tamaño de la imagen con VGG19 y devolver como matriz numpy
train_imgs = load_images(train_dir, 224)

# Seleccionemos una imagen de ejemplo que usaremos para analizar los mapas de características de la CNN
ex_img = train_imgs[0]
plt.imshow(ex_img/255)
plt.axis('off')
plt.show()


def build_autoencoder(input_shape):
    """
    Construye un autoencoder convolucional.
    Args:
    -----------
    input_shape : tuple
        La forma de la entrada de la imagen (altura, ancho, canales).
    Returns:
    -----------
    autoencoder : keras.Model
        El modelo de autoencoder compilado.
    El autoencoder consta de dos partes:
    1. Encoder: Reduce la dimensionalidad de la imagen de entrada.
    2. Decoder: Reconstruye la imagen a partir de la representación comprimida.
    La arquitectura del encoder incluye varias capas Conv2D y MaxPooling2D.
    La arquitectura del decoder incluye varias capas Conv2D y UpSampling2D.
    El modelo se compila con el optimizador 'adam' y la función de pérdida 'binary_crossentropy'.
    """
    input_img = Input(shape=input_shape)

    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(encoded)

    # Decoder
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return autoencoder

# Definir el autoencoder con la forma de las imágenes de entrada
input_shape = (224, 224, 3)  # Asumiendo que las imágenes tienen tamaño 224x224 y 3 canales (RGB)
autoencoder = build_autoencoder(input_shape)

# Mostrar el resumen del modelo
autoencoder.summary()

# Convertir la lista de imágenes a un array de numpy
train_imgs_array = np.array(train_imgs)

# Normalizar las imágenes
train_imgs_array = train_imgs_array / 255.0

# Entrenar el autoencoder
autoencoder.fit(train_imgs_array, train_imgs_array, epochs=50, batch_size=32, shuffle=True)

# Save the autoencoder model
autoencoder.save('autoencoder_model.keras')

# Cargar las imágenes desde train_dir
image_size = 224  # Asumiendo que las imágenes tienen tamaño 224x224
image_files = load_images(train_dir, image_size)
image_filenames = os.listdir(train_dir)

# Extraer etiquetas de los nombres de los archivos
labels = [file.split('_')[1].split('.')[0] for file in image_filenames]

num_images = 5
idx_random = np.random.choice(len(image_files), num_images)

# Cargar las imágenes seleccionadas aleatoriamente
selected_images = [train_imgs[idx] for idx in idx_random]

# Normalizar las imágenes seleccionadas
selected_images = [img / 255.0 if img.max() > 1 else img for img in selected_images]

# Predecir con el autoencoder
y_hat = autoencoder.predict(np.array(selected_images))

# Normalizar las imágenes predichas
y_hat = [img / 255.0 if img.max() > 1 else img for img in y_hat]

fig = plt.figure(figsize=(4, num_images * 3))
fig.subplots_adjust(hspace=0.2, wspace=0.2)
for i in range(num_images):
    ax1 = fig.add_subplot(6, 2, i * 2 + 1)
    ax1.imshow(selected_images[i])
    ax1.set_title(labels[idx_random[i]])
    ax1.axis('off')
    ax2 = fig.add_subplot(6, 2, i * 2 + 2)
    ax2.imshow(y_hat[i])
    ax2.set_title(labels[idx_random[i]])
    ax2.axis('off')
plt.show()