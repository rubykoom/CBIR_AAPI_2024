# CAMBIAR LOS SIGUIENTES DIRECTORIOS DEPENDIENDO DE LA UBICACIÓN DE LOS DATOS

target_directory = "directorio de destino de los datos preprocesados"

import os
import numpy as np
import pandas as pd
import cv2
import faiss
from tensorflow.keras.preprocessing import image as imageprep
from sklearn.preprocessing import normalize
import tensorflow as tf
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
ex_img = train_imgs[1]
plt.imshow(ex_img/255)
plt.axis('off')
plt.show()



import numpy as np

def extract_features_hist(images: np.array) -> np.array:
    N = images.shape[0]  # Número de imágenes
    # Creamos una matriz vacía para almacenar los histogramas
    feats = np.zeros((N, 256 * 3))  # [N, 256 para cada canal R, G, B]
    
    for i in range(N):
        img = images[i]
        # Calculamos histogramas para cada canal (R, G, B)
        hist_r, _ = np.histogram(img[:, :, 0], bins=256, range=(0, 256))
        hist_g, _ = np.histogram(img[:, :, 1], bins=256, range=(0, 256))
        hist_b, _ = np.histogram(img[:, :, 2], bins=256, range=(0, 256))
        
        # Concatenamos los tres histogramas en un solo vector
        feats[i, :] = np.concatenate([hist_r, hist_g, hist_b])
    
    return feats

# Suponiendo que x_train y x_test son tus conjuntos de imágenes
train_imgs_array = np.array(train_imgs)
train_feats_hist = extract_features_hist(train_imgs_array)  # Conjunto de entrenamiento  # Conjunto de prueba

# Asegurarse de que train_feats sea 2D
train_feats_reshaped = train_feats_hist.reshape(train_feats_hist.shape[0], -1)
# Normalizar las características
train_feats_hist = normalize(train_feats_reshaped, norm='l2')



# Convertir las características de entrenamiento a tipo float32
train_feats = train_feats_hist.astype('float32')

# Normalizar los vectores de características usando FAISS
faiss.normalize_L2(train_feats)

# Asegurarse de que train_feats sea 2D
train_feats_reshaped = train_feats.reshape(train_feats.shape[0], -1)

# Construir el índice con Faiss
index = faiss.IndexFlatL2(train_feats_reshaped.shape[1])  # L2 distance
index.add(train_feats_reshaped)  # Añadir las características al índice

# Guardar el índice en un archivo
faiss.write_index(index, 'feat_extract_5')



test_dir = target_directory + '/test'

test_imgs = load_images(test_dir, 224)

# Visualizar una imagen del conjunto de test
test_img = test_imgs[0]



plt.imshow(test_img/255)
plt.axis('off')
plt.show()



def model_feature_extractor5(image):
    """
    Extrae el vector de características de una imagen utilizando el histograma de color.

    Args:
        image (numpy.ndarray): Imagen de entrada.

    Returns:
        numpy.ndarray: Vector de características extraído.
    """
    
    # Redimensionar la imagen
    # img_query = test_img.resize((224, 224))
    image = np.asarray(test_img)
    
    # Asegurarse de que la imagen tenga 3 canales (RGB)
    if image.shape[-1] != 3:
        # Convertir a 3 canales si es necesario
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Añadir la dimensión del lote
    image = np.expand_dims(image, axis=0)
    
    feature_vector = extract_features_hist(image)
    return feature_vector

feature_vector = model_feature_extractor5(test_img)



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



vector = np.float32(feature_vector)

faiss.normalize_L2(vector)

# Buscar las k imágenes más cercanas en el índice FAISS
_, indices = index.search(vector, 5)

# Extraer las imágenes correspondientes a los índices obtenidos
similar_test_images = [train_imgs[idx] for idx in indices[0]]

# Función para visualizar las imágenes más similares con nombres
def plot_similar_images(similar_images, query_image, image_names):
    """
    Dado un conjunto de imágenes similares y la imagen de consulta, las visualiza con sus nombres.
    
    Args:
    - similar_images: Lista de las imágenes más similares
    - query_image: La imagen de consulta
    - image_names: Lista de nombres de las imágenes similares
    """
    k = len(similar_images)  # Número de imágenes similares
    
    # Plotear la imagen de consulta
    plt.figure(figsize=(15, 5))
    plt.subplot(1, k+1, 1)
    plt.imshow(query_image / 255)
    plt.title("Imagen de consulta")
    plt.axis('off')
    
    # Plotear las imágenes más similares con sus nombres
    for i, (sim_img, name) in enumerate(zip(similar_images, image_names)):
        plt.subplot(1, k+1, i+2)
        plt.imshow(sim_img / 255)
        plt.title(f"{name}")
        plt.axis('off')
    
    plt.show()

# Obtener los nombres de los archivos de imagen correspondientes a los índices obtenidos
image_names = get_image_names_by_indices(indices[0], train_dir)

# Visualizar las imágenes más similares con nombres
plot_similar_images(similar_test_images, test_img, image_names)



