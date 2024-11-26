import time
import torch
import faiss
import pathlib
from PIL import Image
import numpy as np
import pandas as pd
import os
import time

import streamlit as st
from streamlit_cropper import st_cropper

from tensorflow.keras.applications import vgg19
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image as imageprep
from sklearn.feature_extraction.image import extract_patches_2d
import cv2
from scipy.cluster.vq import kmeans, vq

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

st.set_page_config(layout="wide")

device = torch.device('cpu')

FILES_PATH = str(pathlib.Path().resolve())

# Path in which the images should be located
IMAGES_PATH = os.path.join(FILES_PATH, 'images')
# Path in which the database should be located
DB_PATH = os.path.join(FILES_PATH, 'database')

DB_FILE = 'db.csv' # name of the database

def preprocess_img_cnn(img_query):
    """
    Preprocesa una imagen para ser utilizada con un modelo CNN.

    Args:
        img_query (PIL.Image): Imagen de consulta.

    Returns:
        numpy.ndarray: Imagen preprocesada con dimensión de lote añadida.
    """
    # Resize and preprocess the image
    img_query = img_query.resize((224, 224))
    image = np.asarray(img_query)
    
    # Ensure the image has 3 channels
    if image.ndim == 2:
        image = np.stack((image,) * 3, axis=-1) 
    
    image = image[:, :, :3]  # Keep only the first 3 channels if there are more
    
    image_prepos = preprocess_input(image)
    image_def = np.expand_dims(image_prepos, axis=0)  # Add batch dimension
    
    return image_def


def get_image_list():
    """
    Obtiene la lista de imágenes desde el archivo CSV de la base de datos.

    Returns:
        list: Lista de nombres de imágenes.
    """
    df = pd.read_csv(os.path.join(DB_PATH, DB_FILE))
    
    # Ensure that the column name is correct
    if 'images' in df.columns:
        image_list = list(df['images'].values)
    else:
        raise ValueError("The 'images' column was not found in the CSV file.")
    return image_list

def model_feature_extractor1(img_query):
    """
    Extrae características de una imagen utilizando un modelo CNN (VGG19).

    Args:
        img_query (PIL.Image): Imagen de consulta.

    Returns:
        numpy.ndarray: Vector de características extraído.
    """
    # Load the model
    vgg19_model = vgg19.VGG19(weights='imagenet')
    base_model = Model(inputs=vgg19_model.inputs, outputs=vgg19_model.get_layer('block5_conv4').output)
    
    # Preprocess the image
    img = preprocess_img_cnn(img_query)

    # Get feature maps
    feature_maps = base_model.predict(img)
    feature_vector = feature_maps.reshape(feature_maps.shape[0], -1)
    return feature_vector


def get_patches(img_file, random_state, patch_size, n_patches=250):
    """
    Extrae subimágenes (parches) de una imagen.

    Args:
        img_file (numpy.ndarray): Imagen de la cual extraer parches.
        random_state (numpy.random.RandomState): Estado aleatorio para la extracción.
        patch_size (tuple): Tamaño de cada parche.
        n_patches (int): Número de parches a extraer.

    Returns:
        numpy.ndarray: Parches extraídos y aplanados.
    """
    # Extract subimages
    patch = extract_patches_2d(img_file, 
                            patch_size=patch_size,
                            max_patches=n_patches, 
                            random_state=random_state)
    
    return patch.reshape((patch.shape[0], -1))  # Flatten each patch

def model_feature_extractor2(img_query):
    """
    Extrae características de una imagen utilizando el modelo Bag of Visual Words.

    Args:
        img_query (PIL.Image): Imagen de consulta.

    Returns:
        list: Lista de vectores de frecuencia de palabras visuales.
    """
    # Load the model
    codebook = np.load('codebook.npy')
    
    orb = cv2.ORB_create()
    
    # Resize and preprocess the image
    img_query = img_query.resize((224, 224))
    image = np.asarray(img_query)
    
    patch_size = (30,30)
    patches = []
    patches.append(get_patches(image, np.random.RandomState(0), patch_size))
    
    descriptors = []

    for img in patches:
        # extract keypoints and descriptors for each image
        _, img_descriptors = orb.detectAndCompute(img, None)
        descriptors.append(img_descriptors)
    
    visual_words = []
    for img_descriptors in descriptors:
        # for each image, map each descriptor to the nearest codebook entry
        img_visual_words, _ = vq(img_descriptors, codebook)
        visual_words.append(img_visual_words)
        
    frequency_vectors = []

    for img_visual_words in visual_words:
    
        # create a frequency vector for each image
        img_frequency_vector = np.zeros(250)
        for word in img_visual_words:
            img_frequency_vector[word] += 1
        frequency_vectors.append(img_frequency_vector)
    
    return frequency_vectors

def model_feature_extractor3(img_query):
    """
    Extrae características de una imagen utilizando un modelo Autoencoder.

    Args:
        img_query (PIL.Image): Imagen de consulta.

    Returns:
        numpy.ndarray: Vector de características extraído.
    """
    filename = 'autoencoder_model.keras'
    autoencoder = load_model(filename)
    # Load the model
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('conv2d_5').output)
    
    # Resize image
    img_query = img_query.resize((224, 224))
    image = np.asarray(img_query)
    
    # Ensure that the image has 3 channels (RGB)
    if image.shape[-1] != 3:
        # Convert to 3 channels if necessary
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    # Get the feature vector
    feature_vector = encoder.predict(image)

    return feature_vector

def model_feature_extractor4(img_query):
    """
    Extrae características de una imagen utilizando los píxeles en bruto.

    Args:
        img_query (PIL.Image): Imagen de consulta.

    Returns:
        numpy.ndarray: Vector de características extraído.
    """
    # Resize the image
    img_query = img_query.resize((224, 224))
    image = np.asarray(img_query)
    
    # Ensure the image has 3 channels (RGB)
    if image.shape[-1] != 3:
        # Convert to 3 channels if necessary
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    # Ensure the image is float32 and normalized (values between 0 and 1)
    image = image.astype(np.float32) / 255.0
    N = image.shape[0]  # Number of images
    feats = image.reshape(N, -1)  # Flatten each image from [224, 224, 3] to [224*224*3]
    
    return feats

def model_feature_extractor5(img_query):
    """
    Extrae características de una imagen utilizando histogramas de color.

    Args:
        img_query (PIL.Image): Imagen de consulta.

    Returns:
        numpy.ndarray: Vector de características extraído.
    """
    # Resize the image
    img_query = img_query.resize((224, 224))
    image = np.asarray(img_query)
    
    # Ensure the image has 3 channels (RGB)
    if image.shape[-1] != 3:
        # Convert to 3 channels if necessary
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    N = image.shape[0]  # Number of images
    # Create an empty array to store histograms
    feats = np.zeros((N, 256 * 3))  # [N, 256 for each channel R, G, B]
    
    for i in range(N):
        img = image[i]
        # Calculate histograms for each channel (R, G, B)
        hist_r, _ = np.histogram(img[:, :, 0], bins=256, range=(0, 256))
        hist_g, _ = np.histogram(img[:, :, 1], bins=256, range=(0, 256))
        hist_b, _ = np.histogram(img[:, :, 2], bins=256, range=(0, 256))
        
        # Concatenate the three histograms into a single vector
        feats[i, :] = np.concatenate([hist_r, hist_g, hist_b])
    
    return feats

def retrieve_image(img_query, feature_extractor, n_imgs=12):
    """
    Recupera imágenes similares a una imagen de consulta utilizando un extractor de características específico.

    Args:
        img_query (PIL.Image): Imagen de consulta.
        feature_extractor (str): Nombre del extractor de características a utilizar.
        n_imgs (int): Número de imágenes a recuperar.

    Returns:
        list: Lista de índices de las imágenes recuperadas.
    """
    if feature_extractor == 'Feature map CNN':
        embeddings = model_feature_extractor1(img_query)
        indexer = faiss.read_index(os.path.join(DB_PATH, 'feat_extract_1'))

    elif feature_extractor == 'Bag of Visual Words':
        embeddings = model_feature_extractor2(img_query)
        indexer = faiss.read_index(os.path.join(DB_PATH, 'feat_extract_2'))

    elif feature_extractor == 'Autoencoder':
        # Specific preprocessing for autoencoder
        embeddings = model_feature_extractor3(img_query)
        embeddings = embeddings.reshape(1, -1)
        indexer = faiss.read_index(os.path.join(DB_PATH, 'feat_extract_3'))

        # Convert embeddings to float32 and ensure they have 2 dimensions
        embeddings = np.array(embeddings).reshape(1, -1)

    elif feature_extractor == 'Raw pixels':
        embeddings = model_feature_extractor4(img_query)
        indexer = faiss.read_index(os.path.join(DB_PATH, 'feat_extract_4'))

    elif feature_extractor == 'Histogram':
        embeddings = model_feature_extractor5(img_query)
        indexer = faiss.read_index(os.path.join(DB_PATH, 'feat_extract_5'))

    vector = np.float32(embeddings)
    
    if feature_extractor != 'Autoencoder':
        faiss.normalize_L2(vector)

    # Perform search in the index with preprocessed vectors
    _, indices = indexer.search(vector, k=n_imgs)

    # Return the found indices
    return indices[0]

def calculate_precision_k(img_query, retriev, k_values=[2, 5, 7, 10]):
    """
    Calcula la métrica Precision@K para una imagen de consulta.

    Args:
        img_query (str): Nombre de la imagen de consulta.
        retriev (list): Lista de índices de las imágenes recuperadas.
        k_values (list): Lista de valores k para los cuales calcular la precisión.

    Returns:
        dict: Diccionario con los valores de Precision@K para cada k.
    """
    img_query_name = img_query.split('_')[1].split('.')[0]
        
    # Get the list of images from the CSV file
    image_list = get_image_list()
        
    # Count number of relevant images retrieved for each k
    precision_results = {}
    for k in k_values:
        if k > len(retriev):
            continue  # Skip if k is greater than number of retrieved images
        retrieved_image_names = [image_list[retriev[i]].split('_')[1].split('.')[0] for i in range(k)]

        # Count how many relevant images are in the retrieved images
        relevant_count = sum(1 for img in retrieved_image_names if img == img_query_name)
            
        # Calculate precision (divide by k to get value between 0 and 1)
        precision = relevant_count / k
        precision_results[k] = round(precision, 2)
        
    return precision_results

def calculate_recall_k(img_query, retriev, k_values=[2, 5, 7, 10]):
    """
    Calcula la métrica Recall@K para una imagen de consulta.

    Args:
        img_query (str): Nombre de la imagen de consulta.
        retriev (list): Lista de índices de las imágenes recuperadas.
        k_values (list): Lista de valores k para los cuales calcular el recall.

    Returns:
        dict: Diccionario con los valores de Recall@K para cada k.
    """
    img_query_name = img_query.split('_')[1].split('.')[0]
    
    # Get the list of images from the CSV file
    image_list = get_image_list()
    
    # Count total number of relevant images (all images with same class as query)
    total_relevant = sum(1 for i in retriev if image_list[i].split('_')[1].split('.')[0] == img_query_name)

    # Calculate recall for each k value
    recall_results = {}
    for k in k_values:
        if k > len(retriev):
            continue

        # Get names of retrieved images up to k
        retrieved_image_names = [image_list[retriev[i]].split('_')[1].split('.')[0] for i in range(k)]

        # Count relevant images found in first k results 
        relevant_found = sum(1 for img in retrieved_image_names if img == img_query_name)

        # Check total_relevant is not zero to avoid division by zero
        if total_relevant == 0:
            recall = 0
        else:
            # Calculate recall as relevant found / total relevant
            recall = relevant_found / total_relevant

        recall_results[k] = round(recall, 2)
    
    return recall_results

def calculate_fscore_k(img_query, retriev, k_values=[2, 5, 7, 10]):
    """
    Calcula la métrica F-score@K para una imagen de consulta.

    Args:
        img_query (str): Nombre de la imagen de consulta.
        retriev (list): Lista de índices de las imágenes recuperadas. 
        k_values (list): Lista de valores k para los cuales calcular el F-score.

    Returns:
        dict: Diccionario con los valores de F-score@K para cada k.
    """
    precision_results = calculate_precision_k(img_query, retriev, k_values)
    recall_results = calculate_recall_k(img_query, retriev, k_values)
                
    fscore_results = {}
    for k in k_values:
        if k not in precision_results or k not in recall_results:
            continue
                    
        precision = precision_results[k]
        recall = recall_results[k]
                    
        # Check if both precision and recall are 0 to avoid division by zero
        if precision == 0 and recall == 0:
            fscore = 0
        else:
            fscore = 2 * (precision * recall) / (precision + recall)
                        
        fscore_results[k] = round(fscore, 2)
        
    return fscore_results


def main():
    st.title('CBIR IMAGE SEARCH')
    
    col1, col2 = st.columns(2)

    with col1:
        st.header('QUERY')

        st.subheader('Choose feature extractor')
        option = st.selectbox('.', ('Feature map CNN', 'Bag of Visual Words', 'Autoencoder', 'Raw pixels', 'Histogram'))

        st.subheader('Upload image')
        img_file = st.file_uploader(label='.', type=['png', 'jpg'])

        if img_file:
            img = Image.open(img_file)
            cropped_img = st_cropper(img, realtime_update=True, box_color='#FF0004')
            
            # Manipulate cropped image at will
            st.write("Preview")
            _ = cropped_img.thumbnail((150,150))
            st.image(cropped_img)

    with col2:
        st.markdown('# RESULTS')
        if img_file:
            image_name = img_file.name.split('_')[1].split('.')[0]
            st.markdown(f'## SIMILAR IMAGES TO {image_name.upper()}')
            placeholder = st.empty()
            placeholder.markdown('**Please wait, retrieving...**')
            start = time.time()

            retriev = retrieve_image(cropped_img, option, n_imgs=12)
            image_list = get_image_list()

            end = time.time()
            time.sleep(1)
            placeholder.markdown('**Finish in ' + str(round(end - start, 2)) + ' seconds**')
            
            # Evaluation metrics
            st.markdown('## Evaluation metrics')
            
            # Create three columns for metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)

            # Precision@K in first column
            with metric_col1:
                st.markdown('### Precision@K')
                precision_results = calculate_precision_k(img_file.name, retriev)
                for k, precision in precision_results.items():
                    st.markdown(f'Precision@{k} = {precision}')

            # Recall@K in second column
            with metric_col2:
                st.markdown('### Recall@K')
                recall_results = calculate_recall_k(img_file.name, retriev)
                for k, recall in recall_results.items():
                    st.markdown(f'Recall@{k} = {recall}')
                
            # F-score@K in third column
            with metric_col3:
                st.markdown('### F-score@K')
                fscore_results = calculate_fscore_k(img_file.name, retriev)
                for k, fscore in fscore_results.items():
                    st.markdown(f'F-score@{k} = {fscore}')

            col3, col4 = st.columns(2)

            with col3:
                image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[0]]))
                st.image(image, use_column_width='always')

            with col4:
                image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[1]]))
                st.image(image, use_column_width='always')

            col5, col6, col7 = st.columns(3)

            with col5:
                for u in range(2, 12, 3):
                    image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[u]]))
                    st.image(image, use_column_width='always')

            with col6:
                for u in range(3, 12, 3):
                    image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[u]]))
                    st.image(image, use_column_width='always')

            with col7:
                for u in range(4, 12, 3):
                    image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[u]]))
                    st.image(image, use_column_width='always')

if __name__ == '__main__':
    main()
