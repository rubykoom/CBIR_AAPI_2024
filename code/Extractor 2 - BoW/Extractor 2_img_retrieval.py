# CAMBIAR LOS SIGUIENTES DIRECTORIOS DEPENDIENDO DE LA UBICACIÓN DE LOS DATOS

target_directory = "directorio de destino de los datos preprocesados"


import os
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.cluster import KMeans
import cv2
from tensorflow.keras.preprocessing import image_dataset_from_directory
from PIL import Image
import tensorflow as tf
import faiss
from scipy.cluster.vq import kmeans, vq
from tensorflow.keras.preprocessing import image as imageprep

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
    img = Image.open(img_path)
    if img.mode == 'P' and 'transparency' in img.info:
        img = img.convert('RGBA')
    img = img.resize((image_size, image_size))
    return np.array(img)

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



# Cargar el conjunto de datos de imágenes
train = []
fruit_path = os.path.join(target_directory, 'train')
if os.path.exists(fruit_path):
    for img_file in os.listdir(fruit_path):
        img_path = os.path.join(fruit_path, img_file)
        img = image_to_array(img_path, 224)
        train.append(img)

print(f"Total number of images in train: {len(train)}")



def plot_10_by_10_images(images):
    """
    Muestra una cuadrícula de 10x10 imágenes.

    Args:
        images (list): Lista de imágenes a mostrar. Cada imagen debe ser un array numpy.
    """
    # Tamaño de la figura
    fig = plt.figure(figsize=(10,10))

    # Mostrar la cuadrícula de imágenes
    for x in range(10):
        for y in range(10):
            fig.add_subplot(10, 10, 10*y+x+1)
            if 10*y + x < len(images):
                plt.imshow(images[10*y + x])
            else:
                # Imagen en blanco si no hay suficientes imágenes
                plt.imshow(np.zeros((100, 100, 3), dtype=np.uint8))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
    plt.show()



plot_10_by_10_images(train)



def get_patches(img_file, random_state, patch_size, n_patches=250):
    """
    Extrae subimágenes.

    Args:
        img_file (numpy.ndarray): Matriz de la imagen (no la ruta).
        random_state (numpy.random.RandomState): Estado aleatorio para la extracción de parches.
        patch_size (tuple): Tamaño de cada parche.
        n_patches (int): Número de parches a extraer.

    Returns:
        numpy.ndarray: Parches redimensionados a (n_patches, alto_parche * ancho_parche * canales).
    """
    # Extraer subimágenes (retorna (n_patches, alto_parche, ancho_parche, canales))
    patch = extract_patches_2d(img_file, 
                               patch_size=patch_size,
                               max_patches=n_patches, 
                               random_state=random_state)
    
    # Redimensionar a (n_patches, alto_parche * ancho_parche * canales)
    return patch.reshape((n_patches, -1))  # Aplanar cada parche



patch_size = (30,30)

patches = []
for i in range(0, len(train)):
    patches.append(get_patches(train[i], np.random.RandomState(0), patch_size))



print('Patches extracted to create dictionary of features')
print('Total of images = ', len(patches))
print('Size of each array of patches = ', patches[0].shape)

img_ind = 32
plt.figure(figsize=(8,3))
for i in np.arange(1,11):
    plt.subplot(2,5,i)
    plt.imshow(patches[img_ind][i].reshape((patch_size[0],patch_size[1],3)))
    plt.axis('off')



orb = cv2.ORB_create()

# Inicializar listas donde almacenaremos *todos* los descriptores
descriptors = []

for img in patches:
    # Extraer puntos clave y descriptores para cada imagen
    _, img_descriptors = orb.detectAndCompute(img, None)
    descriptors.append(img_descriptors)



all_descriptors = []

# Extraer listas de descriptores de imágenes
for img_descriptors in descriptors:
    
    # Extraer descriptores específicos dentro de la imagen
    for descriptor in img_descriptors:
        all_descriptors.append(descriptor)
        
# Convertir a un solo array numpy
all_descriptors = np.stack(all_descriptors)



all_descriptors.shape



all_descriptors = np.float32(all_descriptors)



# Realizar la agrupación k-means para construir el codebook
k = 250
# codebook, _ = kmeans(all_descriptors, k)



# Guardar el codebook en un archivo .npy
# np.save('codebook.npy', codebook)



# Cargar el codebook desde un archivo .npy
codebook = np.load('codebook.npy')



visual_words = []
for img_descriptors in descriptors:
    
    # Para cada imagen, asignar cada descriptor a la entrada más cercana del codebook
    img_visual_words, distance = vq(img_descriptors, codebook)
    visual_words.append(img_visual_words)



# Veamos los visual words de la imagen 200
visual_words[199][:5], len(visual_words[199])



frequency_vectors = []

for img_visual_words in visual_words:
    # Crear un vector de frecuencia para cada imagen
    img_frequency_vector = np.zeros(k)
    for word in img_visual_words:
        img_frequency_vector[word] += 1
    frequency_vectors.append(img_frequency_vector)
    
# Apilar juntos en un array numpy
frequency_vectors = np.stack(frequency_vectors)



plt.bar(list(range(k)), frequency_vectors[0])
plt.show()



# Convertir a float32
vectors = np.array(frequency_vectors, dtype=np.float32)

# Normalizar los vectores de características usando FAISS
faiss.normalize_L2(vectors)

# Obtener la dimensión de las características
feature_dim = vectors.shape[1]

# Crear un índice FAISS basado en la distancia L2
index = faiss.IndexFlatL2(feature_dim)

# Añadir los vectores de características al índice
index.add(vectors)

# Guardar el índice en un archivo
faiss.write_index(index, "feat_extract_2")



test_dir = target_directory + '/test'

test_imgs = load_images(test_dir, 224)

# Visualizar una imagen del conjunto de test
test_img = test_imgs[8]



plt.imshow(test_img/255)
plt.axis('off')
plt.show()



def model_feature_extractor2(img_query):
    # Cargar el codebook desde un archivo .npy
    codebook = np.load('codebook.npy')
    
    orb = cv2.ORB_create()
    
    image = np.asarray(img_query)
    
    patch_size = (30,30)
    patches = []
    patches.append(get_patches(image, np.random.RandomState(0), patch_size))
    
    descriptors = []

    for img in patches:
        # Extraer descriptores para cada imagen
        _, img_descriptors = orb.detectAndCompute(img, None)
        descriptors.append(img_descriptors)
    
    visual_words = []
    for img_descriptors in descriptors:
        # Para cada imagen, asignar cada descriptor a la entrada más cercana del codebook
        img_visual_words, _ = vq(img_descriptors, codebook)
        visual_words.append(img_visual_words)
        
    frequency_vectors = []

    for img_visual_words in visual_words:
    
        # Crear un vector de frecuencia para cada imagen
        img_frequency_vector = np.zeros(250)
        for word in img_visual_words:
            img_frequency_vector[word] += 1
        frequency_vectors.append(img_frequency_vector)
    
    return frequency_vectors



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



feature_vector = model_feature_extractor2(test_img)

vector = np.float32(feature_vector)
faiss.normalize_L2(vector)

# Buscar las k imágenes más cercanas en el índice FAISS
_, indices = index.search(vector, 5)

# Extraer las imágenes correspondientes a los índices obtenidos
similar_test_images = [train[idx] for idx in indices[0]]

# Función para visualizar las imágenes más similares
def plot_similar_images(similar_images, query_image, indices, image_dir):
    """
    Dado un conjunto de imágenes similares y la imagen de consulta, las visualiza junto con sus nombres de archivo.
    
    Parámetros:
    - similar_images: Lista de las imágenes más similares
    - query_image: La imagen de consulta
    - indices: Lista de índices de las imágenes similares
    - image_dir: Directorio donde se encuentran las imágenes
    """
    k = len(similar_images)  # Número de imágenes similares
    
    # Obtener los nombres de los archivos de imagen
    image_names = get_image_names_by_indices(indices, image_dir)
    
    # Plotear la imagen de consulta
    plt.figure(figsize=(15, 5))
    plt.subplot(1, k+1, 1)
    plt.imshow(query_image / 255)
    plt.title("Imagen de Consulta")
    plt.axis('off')
    
    # Plotear las imágenes más similares
    for i, (sim_img, img_name) in enumerate(zip(similar_images, image_names)):
        plt.subplot(1, k+1, i+2)
        plt.imshow(sim_img / 255)
        plt.title(f"{img_name}")
        plt.axis('off')
    
    plt.show()

# Visualizar las imágenes más similares
plot_similar_images(similar_test_images, test_img, indices[0], fruit_path)


