# Instalación:

En windows:<br/>
- Instalar el el framework Anaconda, la version de python 3.6<br/>
<https://www.anaconda.com/download/#linux<br/>
- Instalar la libreria de tensorflow (para Python)<br/>
https://www.tensorflow.org/install/<br/>
<br/>
- Instalar keras y cv2 con pip:<br/>
(En la consola de windows)<br/>
pip install keras<br/>
pip install cv2<br/>
<br/>
Descargar los datos de entrenamiento de la red en: https://www.kaggle.com/c/dogs-vs-cats/data<br/>
<br/>
Crear la siguiente estructura de carpetas una vez descargados los datos:<br/>
data/<br/>
    train/<br/>
        dogs/<br/>
            dog001.jpg<br/>
            dog002.jpg<br/>
            ...<br/>
        cats/<br/>
            cat001.jpg<br/>
            cat002.jpg<br/>
            ...<br/>
    validation/<br/>
        dogs/<br/>
            dog001.jpg<br/>
            dog002.jpg<br/>
            ...<br/>
        cats/<br/>
            cat001.jpg<br/>
            cat002.jpg<br/>
            ...<br/>
            <br/>
# Para correr la red:<sbr/>
python red-neuronal.py<br/>
