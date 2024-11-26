
# CAMBIAR LOS SIGUIENTES DIRECTORIOS DEPENDIENDO DE LA UBICACIÓN DE LOS DATOS

root_directory = "directorio original de los datos"
target_directory = "directorio de destino de los datos preprocesados"


import os
import shutil
import hashlib
from PIL import Image
import pandas as pd
from bing_image_downloader import downloader



def filter_and_rename_files(root_dir, target_dir, sub_dirs, max_fruits=30):
    """
    Filtra y renombra archivos de imágenes en subdirectorios específicos.

    Args:
        root_dir (str): Directorio raíz que contiene las divisiones de datos ('train', 'test', 'validation').
        target_dir (str): Directorio de destino donde se guardarán las imágenes filtradas y renombradas.
        sub_dirs (list): Lista de subdirectorios de frutas a procesar.
        max_fruits (int, optional): Número máximo de archivos de frutas a mantener en 'train'. Por defecto 30.
    """
    # Crear el directorio de destino si no existe
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Iterar sobre las divisiones de datos: 'train', 'test', 'validation'
    for split in ['train', 'test', 'validation']:
        split_dir = os.path.join(root_dir, split)
        target_split_dir = os.path.join(target_dir, split)
        
        # Crear el directorio de destino para la división si no existe
        if not os.path.exists(target_split_dir):
            os.makedirs(target_split_dir)
        
        # Iterar sobre los subdirectorios de frutas
        for sub_dir in sub_dirs:
            src_dir = os.path.join(split_dir, sub_dir)
            
            if os.path.exists(src_dir):
                files = os.listdir(src_dir)
                # Limitar el número de archivos en 'train' a max_fruits
                if split == 'train':
                    files = files[:max_fruits]
                for idx, file in enumerate(files):
                    file_extension = os.path.splitext(file)[1]
                    new_file_name = f"{str(idx+1).zfill(2)}_{sub_dir.lower()}_prepros{file_extension}"
                    src_file_path = os.path.join(src_dir, file)
                    dst_file_path = os.path.join(target_split_dir, new_file_name)
                    shutil.copy(src_file_path, dst_file_path)
                    
        # Eliminar la carpeta 'validation' sin importar si está vacía o no
        validation_dir = os.path.join(target_dir, 'validation')
        if os.path.exists(validation_dir):
            shutil.rmtree(validation_dir)

# Seleccionamos los subdirectorios de frutas a procesar: 10 frutas
fruits_dirs = ['Banana', 'Apple', 'Pear', 'Grapes', 'Orange', 'Kiwi', 'Watermelon', 'Pomegranate', 'Pineapple', 'Mango']

# Example usage
filter_and_rename_files(root_directory, target_directory, fruits_dirs)



def hash_image(image_path):
    """
    Genera un hash para una imagen dada.

    Args:
        image_path (str): La ruta del archivo de imagen.

    Returns:
        str: El hash MD5 de la imagen.
    """
    with Image.open(image_path) as img:
        # Redimensionar la imagen a 224x224 píxeles y convertirla a RGB
        img = img.resize((224, 224)).convert('RGB')
        # Generar y devolver el hash MD5 de la imagen
        return hashlib.md5(img.tobytes()).hexdigest()

def remove_duplicate_images(train_dir, test_dir):
    """
    Elimina imágenes duplicadas en el conjunto de test a partir del conjunto de train y devuelve un DataFrame con el conteo de eliminaciones por fruta.

    Args:
        train_dir (str): Directorio del conjunto de entrenamiento.
        test_dir (str): Directorio del conjunto de prueba.

    Returns:
        pd.DataFrame: DataFrame con el conteo de eliminaciones por fruta.
    """
    train_hashes = set()  # Conjunto para almacenar los hashes de las imágenes del conjunto de entrenamiento
    eliminations = {}  # Diccionario para contar las eliminaciones por tipo de fruta

    # Generar hashes para las imágenes del conjunto de entrenamiento
    for filename in os.listdir(train_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(train_dir, filename)
            image_hash = hash_image(image_path)
            train_hashes.add(image_hash)
    
    # Comprobar y eliminar imágenes duplicadas en el conjunto de prueba
    for filename in os.listdir(test_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(test_dir, filename)
            image_hash = hash_image(image_path)
            if image_hash in train_hashes:
                os.remove(image_path)
                # Asumiendo que el nombre de la fruta está al principio del nombre del archivo
                fruit_type = filename.split('_')[1].split('.')[0]
                if fruit_type in eliminations:
                    eliminations[fruit_type] += 1
                else:
                    eliminations[fruit_type] = 1
                print(f"Imagen duplicada eliminada: {filename}")

    # Convertir el diccionario de eliminaciones a un DataFrame
    eliminations_df = pd.DataFrame([(fruit, int(filenames)) for fruit, filenames in eliminations.items()],
                                columns=['Fruta', 'Total'])
    eliminations_df.set_index('Fruta', inplace=True)
    
    return eliminations_df

# Aplicación
train_dir = target_directory + '/train'
test_dir = target_directory + '/test'
eliminations_df = remove_duplicate_images(train_dir, test_dir)
print(eliminations_df)



def download_and_add_images(eliminations_df, test_dir):
    """
    Busca en internet (BING) nuevas imágenes y las añade a la carpeta test.
    
    Args:
        eliminations_df (pd.DataFrame): DataFrame con el conteo de eliminaciones por fruta.
        test_dir (str): Directorio donde se añadirán las nuevas imágenes.
    """
    for fruit, row in eliminations_df.iterrows():
        num_eliminated = int(row['Total'])  # Asegurarse de que num_eliminated sea un entero
        search_query = fruit + " fruit"
        
        # Descargar imágenes
        downloader.download(search_query, limit=num_eliminated, output_dir=test_dir, adult_filter_off=True, force_replace=False, timeout=60)
        
        # Renombrar y mover imágenes descargadas
        downloaded_images_dir = os.path.join(test_dir, search_query)
        downloaded_images = os.listdir(downloaded_images_dir)
        
        for idx, img_name in enumerate(downloaded_images):
            new_file_name = f"{str(num_eliminated + idx + 1).zfill(2)}_{fruit.lower()}.jpg"
            new_file_path = os.path.join(test_dir, new_file_name)
            # Mover la imagen descargada a la carpeta test con el nuevo nombre
            shutil.move(os.path.join(downloaded_images_dir, img_name), new_file_path)
            print(f"Imagen añadida: {new_file_path}")

# Aplicación
download_and_add_images(eliminations_df, test_dir)



def move_and_rename_images(test_dir):
    """
    Mueve y renombra las imágenes en la carpeta test.
    
    Args:
        test_dir (str): Directorio donde se encuentran las imágenes.
    """
    # Obtener las carpetas de frutas en el directorio test
    fruit_folders = os.listdir(test_dir)
    fruit_folders = [folder for folder in fruit_folders if os.path.isdir(os.path.join(test_dir, folder))]
    
    for fruit in fruit_folders:
        fruit_path = os.path.join(test_dir, fruit)
        images = os.listdir(fruit_path)
        
        for idx, img_name in enumerate(images):
            img_path = os.path.join(fruit_path, img_name)
            new_file_name = f"{str(idx + 1).zfill(2)}_{fruit.lower()}.jpg"
            new_file_path = os.path.join(test_dir, new_file_name)
            
            # Mover y renombrar la imagen
            shutil.move(img_path, new_file_path)
        
        # Eliminar la carpeta vacía
        os.rmdir(fruit_path)
    
    # Renombrar todas las imágenes en la carpeta test
    all_images = os.listdir(test_dir)
    for idx, img_name in enumerate(all_images):
        if '_prepros' in img_name:
            fruit_type = img_name.split('_')[1]
            new_file_name = f"{str(idx + 1).zfill(2)}_{fruit_type}.jpg"
        else:
            fruit_type = img_name.split('_')[1].split(' ')[0]
            new_file_name = f"{str(idx + 1).zfill(2)}_{fruit_type}.jpg"
        
        old_file_path = os.path.join(test_dir, img_name)
        new_file_path = os.path.join(test_dir, new_file_name)
        
        # Mover y renombrar la imagen
        shutil.move(old_file_path, new_file_path)

# Aplicación
move_and_rename_images(test_dir)
move_and_rename_images(train_dir)



def remove_duplicates(train_dir, test_dir):
    """
    Elimina imágenes duplicadas en el conjunto de prueba a partir del conjunto de entrenamiento.
    
    Args:
        train_dir (str): Directorio del conjunto de entrenamiento.
        test_dir (str): Directorio del conjunto de prueba.
        
    Returns:
        int: Número de imágenes duplicadas eliminadas.
    """
    train_hashes = set()  # Conjunto para almacenar los hashes de las imágenes del conjunto de entrenamiento
    duplicates_removed = 0  # Contador de imágenes duplicadas eliminadas

    # Generar hashes para las imágenes del conjunto de entrenamiento
    for filename in os.listdir(train_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(train_dir, filename)
            image_hash = hash_image(image_path)
            train_hashes.add(image_hash)
    
    # Comprobar y eliminar imágenes duplicadas en el conjunto de prueba
    for filename in os.listdir(test_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(test_dir, filename)
            image_hash = hash_image(image_path)
            if image_hash in train_hashes:
                os.remove(image_path)
                duplicates_removed += 1
                print(f"Imagen duplicada eliminada: {filename}")

    print(f"Total de imágenes duplicadas eliminadas: {duplicates_removed}")

# Aplicación
remove_duplicates(train_dir, test_dir)



def remove_duplicates_in_train(train_dir):
    """
    Elimina imágenes duplicadas en el conjunto de entrenamiento.
    
    Args:
        train_dir (str): Directorio del conjunto de entrenamiento.
        
    Returns:
        int: Número de imágenes duplicadas eliminadas.
    """
    image_hashes = set()  # Conjunto para almacenar los hashes de las imágenes
    duplicates_removed = 0  # Contador de imágenes duplicadas eliminadas

    # Comprobar y eliminar imágenes duplicadas en el conjunto de entrenamiento
    for filename in os.listdir(train_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(train_dir, filename)
            image_hash = hash_image(image_path)  # Calcular el hash de la imagen
            if image_hash in image_hashes:
                os.remove(image_path)  # Eliminar la imagen si el hash ya existe
                duplicates_removed += 1
                print(f"Imagen duplicada eliminada: {filename}")
            else:
                image_hashes.add(image_hash)  # Añadir el hash al conjunto

    print(f"Total de imágenes duplicadas eliminadas: {duplicates_removed}")

# Aplicación
remove_duplicates_in_train(train_dir)



def crear_db_csv(carpeta):
    """
    Crea un archivo CSV con una lista de los nombres de archivos en la carpeta especificada.
    Args:
        carpeta (str): La ruta de la carpeta que contiene los archivos.
    Returns:
        None: La función guarda un archivo CSV llamado 'db.csv' en el directorio actual.
    """
    # Obtener la lista de archivos en la carpeta
    archivos = os.listdir(carpeta)
    
    # Crear un DataFrame con una columna llamada 'images'
    df = pd.DataFrame(archivos, columns=['images'])
    
    # Guardar el DataFrame en un archivo CSV
    df.to_csv('db.csv', index=False)

# Llamar a la función con la carpeta 'train'
crear_db_csv(target_directory + '/train')



