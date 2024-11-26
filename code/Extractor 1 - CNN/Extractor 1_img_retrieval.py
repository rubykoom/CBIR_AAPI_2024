# CAMBIAR LOS SIGUIENTES DIRECTORIOS DEPENDIENDO DE LA UBICACIÓN DE LOS DATOS

target_directory = "directorio de destino de los datos preprocesados"

import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image as imageprep
from tensorflow.keras.applications import vgg19
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import preprocess_input
import faiss

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
iconic_imgs = load_images(train_dir, 224)

# Seleccionemos una imagen de ejemplo que usaremos para analizar los mapas de características de la CNN
ex_img = iconic_imgs[0]
plt.imshow(ex_img/255)
plt.axis('off')
plt.show()

# Cargar el modelo VGG19 preentrenado
vgg19_model = vgg19.VGG19(weights='imagenet')

vgg19_model.summary()


# Define un nuevo modelo para obtener el mapa de características de la capa block5_conv4
b5c4_model = Model(inputs=vgg19_model.inputs, outputs=vgg19_model.get_layer('block5_conv4').output)

# Añadir dimensión y preprocesar para escalar los valores de los píxeles para VGG
ex_img = np.expand_dims(ex_img, axis=0)
ex_img = preprocess_input(ex_img)

# Obtener el mapa de características
b5c4_feature_map = b5c4_model.predict(ex_img)

# Validar dimensiones
b5c4_feature_map.shape


def plot_feature_map(feature_map, max_grid):
    """
    Dibuja una cuadrícula de mapas de características a partir de un tensor de mapas de características dado.

    Parámetros:
    feature_map (numpy.ndarray): Un tensor 4D de forma (batch_size, altura, ancho, canales) que representa los mapas de características.
    max_grid (int): El tamaño máximo de la cuadrícula a mostrar. La función mostrará max_grid x max_grid mapas de características.

    Retorna:
    None: Esta función muestra un gráfico y no retorna ningún valor.

    Ejemplo:
    >>> plot_feature_map(feature_map, 4)
    Esto mostrará una cuadrícula de 4x4 mapas de características del tensor feature_map dado.

    Nota:
    - La función asume que el tensor feature_map tiene al menos max_grid**2 canales.
    - La función mostrará los primeros max_grid**2 canales del tensor feature_map.
    """

    fig, ax = plt.subplots(max_grid, max_grid, figsize=(7,7))
    channel_idx = 0

    for i in range(max_grid):
        for j in range(max_grid):
            ax[i][j].imshow(feature_map[0, :, :, channel_idx])
            ax[i][j].axis('off')
            channel_idx += 1

    fig.suptitle(f'Mapa de Características - Mostrando {max_grid**2} de {feature_map.shape[3]} Canales')
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

# Generate visualization
plot_feature_map(b5c4_feature_map, 8)


def get_feature_maps(model, image_holder):
    """
    Obtiene los mapas de características de un modelo dado y una imagen.
    Parámetros:
    model (keras.Model): El modelo preentrenado que se utilizará para extraer los mapas de características.
    image_holder (numpy.ndarray): Un array que contiene la imagen o las imágenes de entrada.
    Retorna:
    numpy.ndarray: Un array que contiene los vectores de características extraídos de los mapas de características del modelo.
    """

    # Añadir dimensión y preprocesar para escalar los valores de los píxeles para VGG
    images = np.asarray(image_holder)
    images = preprocess_input(images)

    # Obtener mapas de características
    feature_maps = model.predict(images)

    # Redimensionar para aplanar el tensor de características en vectores de características
    feature_vector = feature_maps.reshape(feature_maps.shape[0], -1)

    return feature_vector

# Extraer mapas de características de la capa block5_conv4
all_b5c4_features = get_feature_maps(b5c4_model, iconic_imgs)

# Crear índice FAISS para cada conjunto de características, almacenado en un diccionario
index_list = []
features = [all_b5c4_features]

for feature_map in features:
    # Normalizar las características
    faiss.normalize_L2(feature_map)
    feature_dim = feature_map.shape[1]
    index = faiss.IndexFlatL2(feature_dim)
    index.add(feature_map)

    index_list.append(index)

faiss.write_index(index_list[0], "feat_extract_1")



test_dir = target_directory + '/test'

test_imgs = load_images(test_dir, 224)

# Visualizar una imagen del conjunto de test
test_img = test_imgs[0]

plt.imshow(test_img/255)
plt.axis('off')
plt.show()

def preprocess_img_cnn(img_query):

    # Resize and preprocess the image
    # img_query = img_query.resize((224, 224))
    image = np.asarray(img_query)

    # Ensure the image has 3 channels
    if image.ndim == 2:
        image = np.stack((image,) * 3, axis=-1)

    image = image[:, :, :3]  # Keep only the first 3 channels if there are more

    image_prepos = preprocess_input(image)
    image_def = np.expand_dims(image_prepos, axis=0)  # Add batch dimension

    return image_def

def model_feature_extractor1(img_query):
    # Load the model
    vgg19_model = vgg19.VGG19(weights='imagenet')
    base_model = Model(inputs=vgg19_model.inputs, outputs=vgg19_model.get_layer('block5_conv4').output)

    # Preprocess the image
    img = preprocess_img_cnn(img_query)

    # Get feature maps
    feature_maps = base_model.predict(img)
    feature_vector = feature_maps.reshape(feature_maps.shape[0], -1)
    return feature_vector

# Uso de la función
image = np.asarray(test_img)
feature_vector = model_feature_extractor1(image)

def get_image_names_by_indices(indices, image_dir):
    """
    Obtiene los nombres de los archivos de imagen dados una lista de índices.

    Args:
        indices (list): Lista de índices de las imágenes.
        image_dir (str): Directorio donde se encuentran las imágenes.

    Returns:
        list: Lista de nombres de los archivos de imagen.
    """
    image_list = os.listdir(image_dir)
    return [image_list[index].split('_')[1].split('.')[0] for index in indices]

# Buscar las k imágenes más cercanas en el índice FAISS
_, indices = index.search(feature_vector, 5)

# Extraer las imágenes correspondientes a los índices obtenidos
similar_test_images = [iconic_imgs[idx] for idx in indices[0]]

# Función para visualizar las imágenes más similares con sus nombres
def plot_similar_images(similar_images, query_image, image_names):
    """
    Dado un conjunto de imágenes similares y la imagen de consulta, las visualiza con sus nombres.

    Args:
    similar_images (list): Lista de las imágenes más similares.
    query_image (numpy.ndarray): La imagen de consulta.
    image_names (list): Lista de nombres de las imágenes similares.
    """
    k = len(similar_images)  # Número de imágenes similares

    # Plotear la imagen de consulta
    plt.figure(figsize=(15, 5))
    plt.subplot(1, k+1, 1)
    plt.imshow(query_image / 255)
    plt.title("Query Image")
    plt.axis('off')

    # Plotear las imágenes más similares con sus nombres
    for i, (sim_img, name) in enumerate(zip(similar_images, image_names)):
        plt.subplot(1, k+1, i+2)
        plt.imshow(sim_img / 255)
        plt.title(f"{name}")
        plt.axis('off')

    plt.show()

# Obtener los nombres de las imágenes similares
image_names = get_image_names_by_indices(indices[0], train_dir)

# Visualizar las imágenes más similares con sus nombres
plot_similar_images(similar_test_images, test_img, image_names)