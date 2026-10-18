
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras
import os

st.set_page_config(
    page_title="CNN Image Classifier",
    page_icon="🧠"
)

st.title("CNN Image Classifier")
st.write("Klasifikasi gambar menggunakan CNN dan Transfer Learning MobileNetV2.")

MODEL_PATH = "model_transfer_learning.keras"

class_names = [
    "Pesawat", "Mobil", "Burung", "Kucing", "Rusa",
    "Anjing", "Katak", "Kuda", "Kapal", "Truk"
]

@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error("File model tidak ditemukan.")
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Model gagal dimuat: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload gambar untuk diklasifikasikan",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diuji", use_container_width=True)

    if st.button("Klasifikasikan Gambar"):
        img = image.resize((32, 32))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("Sedang melakukan prediksi..."):
            predictions = model.predict(img_array, verbose=0)[0]

        index = int(np.argmax(predictions))

        st.subheader("Hasil Prediksi")
        st.success(f"Kelas: {class_names[index]}")
        st.metric(
            "Confidence",
            f"{float(predictions[index]) * 100:.2f}%"
        )

        st.write("Probabilitas setiap kelas:")
        for i, name in enumerate(class_names):
            st.write(f"{name}: {predictions[i] * 100:.2f}%")
