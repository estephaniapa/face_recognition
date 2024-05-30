import streamlit as st
import numpy as np
from PIL import Image
import json
from tensorflow.keras.models import model_from_json
import cv2

# Función para cargar el modelo de reconocimiento facial
@st.cache_resource
def load_model(model_json_path, weights_path):
    with open(model_json_path, 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights(weights_path)
    return model

class FaceRecognitionApp:
    def __init__(self, model_json_path, weights_path, database_path):
        self.model = load_model(model_json_path, weights_path)
        self.database = self.load_database(database_path)
        self.threshold = 1.0  # Umbral para la verificación

    def load_database(self, database_path):
        with open(database_path, 'r') as f:
            database = json.load(f)
        return {name: np.array(encoding) for name, encoding in database.items()}

    def preprocess_image(self, image):
        image = np.around(np.array(image) / 255.0, decimals=12)
        image = np.expand_dims(image, axis=0)
        embedding = self.model.predict_on_batch(image)
        return embedding / np.linalg.norm(embedding, ord=2)

    def verify_identity(self, image):
        encoding = self.preprocess_image(image)
        min_dist = float('inf')
        identity = None

        for name, db_enc in self.database.items():
            dist = np.linalg.norm(encoding - db_enc)
            if dist < min_dist:
                min_dist = dist
                identity = name

        st.write(f'Distancia mínima: {min_dist}')  # Registro de la distancia mínima
        if min_dist < 0.85:
            return identity
        else:
            return None

def main():
    # Mostrar el logo centrado
    st.image('FaceRecon.png', use_column_width=True)

    # Interactividad: Cargar una imagen para el reconocimiento facial
    st.subheader('Verificar Identidad:')
    uploaded_image = st.file_uploader('Sube tu imagen', type=['jpg', 'jpeg', 'png'])

    if uploaded_image:
        image = Image.open(uploaded_image).convert('RGB')
        # Convertir la imagen PIL a un array NumPy
        image_np = np.array(image)

        # Cambiar el orden de los canales de color de RGB a BGR
        image_np = image_np.transpose((1, 0, 2))

        # Redimensionar la imagen utilizando OpenCV
        resized_image = cv2.resize(image_np, (160, 160))


        st.image(resized_image, caption='Imagen cargada correctamente', use_column_width=True)

        if st.button('Verificar Identidad'):
            with st.spinner('Verificando...'):
                face_recognition_app = FaceRecognitionApp('model/model.json', 'model/model.h5', 'database.json')
                identity = face_recognition_app.verify_identity(resized_image)
                if identity:
                    st.success(f'Bienvenido/a, {identity}!')
                else:
                    st.error('Lo sentimos, no hemos podido confirmar tu identidad.')

if __name__ == '__main__':
    main()
