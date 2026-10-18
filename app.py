
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="CNN Image Classifier",
    page_icon="🧠"
)

st.title("CNN Image Classifier")
st.write(
    "Aplikasi klasifikasi gambar menggunakan "
    "Convolutional Neural Network (CNN)."
)

uploaded_file = st.file_uploader(
    "Upload gambar untuk diklasifikasikan",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah")
    st.success("Gambar berhasil diunggah!")
    st.info("Model CNN akan dihubungkan setelah file model siap.")
