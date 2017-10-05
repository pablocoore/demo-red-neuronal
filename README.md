# Instalación:
En windows:

- Instalar el el framework Anaconda, la version de python 3.6
https://www.anaconda.com/download/#linux

- Instalar la libreria de tensorflow (para Python)
https://www.tensorflow.org/install/

- Instalar keras y cv2 con pip:
(En la consola de windows)
pip install keras
pip install cv2

Descargar los datos de entrenamiento de la red en: https://www.kaggle.com/c/dogs-vs-cats/data

Crear la siguiente estructura de carpetas una vez descargados los datos:
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...

#Correr la red:
python red-neuronal.py
