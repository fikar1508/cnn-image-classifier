
import os
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image
from tensorflow import keras

st.set_page_config(
    page_title="NeuroVision AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# SESSION STATE
# =========================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# THEME
# =========================
dark = st.session_state.theme == "dark"

if dark:
    bg = "#08111F"
    panel = "#101D30"
    text = "#E8F0FC"
    muted = "#91A4BF"
    border = "#243650"
    accent = "#38BDF8"
else:
    bg = "#F3F7FC"
    panel = "#FFFFFF"
    text = "#17243A"
    muted = "#62718A"
    border = "#DCE5F0"
    accent = "#0878D1"

st.markdown(f"""
<style>
.stApp {{
    background: {bg};
    color: {text};
}}
[data-testid="stHeader"] {{
    background: transparent;
}}
[data-testid="stSidebar"] {{
    background: {panel};
    border-right: 1px solid {border};
}}
.block-container {{
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}}
h1, h2, h3, p, label, span {{
    color: {text};
}}
.hero {{
    padding: 28px;
    border: 1px solid {border};
    border-radius: 22px;
    background: linear-gradient(120deg, {panel}, {bg});
    margin-bottom: 24px;
}}
.eyebrow {{
    color: {accent};
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
}}
.hero-title {{
    font-size: clamp(28px, 4vw, 42px);
    font-weight: 800;
    margin: 10px 0;
}}
.hero-desc {{
    color: {muted};
    font-size: 16px;
}}
.panel {{
    background: {panel};
    border: 1px solid {border};
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 16px;
}}
.small-muted {{
    color: {muted};
    font-size: 13px;
}}
div.stButton > button {{
    border-radius: 10px;
    min-height: 42px;
    font-weight: 600;
}}
hr {{
    border-color: {border};
}}
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## 🧠 NeuroVision")
    st.caption("AI Image Classification")
    st.divider()

    st.markdown("### Tampilan")
    theme_choice = st.radio(
        "Pilih tema",
        ["Gelap", "Terang"],
        index=0 if dark else 1,
        label_visibility="collapsed"
    )

    new_theme = "dark" if theme_choice == "Gelap" else "light"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

    st.divider()
    st.markdown("### Tentang Model")
    st.write("**Arsitektur:** MobileNetV2")
    st.write("**Metode:** Transfer Learning")
    st.write("**Dataset:** CIFAR-10")
    st.write("**Jumlah kelas:** 10")
    st.caption("Model klasifikasi gambar berbasis deep learning.")

    st.divider()
    page = st.radio(
        "Navigasi",
        ["Klasifikasi Gambar", "Riwayat Prediksi", "Tentang Proyek"],
        label_visibility="collapsed"
    )

# =========================
# MODEL
# =========================
MODEL_PATH = "model_transfer_learning.keras"

CLASS_NAMES = [
    "Pesawat", "Mobil", "Burung", "Kucing", "Rusa",
    "Anjing", "Katak", "Kuda", "Kapal", "Truk"
]

@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)

st.markdown("""
<div class="hero">
    <div class="eyebrow">COMPUTER VISION • DEEP LEARNING</div>
    <div class="hero-title">NeuroVision AI</div>
    <div class="hero-desc">
        Klasifikasikan gambar dengan model CNN berbasis
        Transfer Learning MobileNetV2.
    </div>
</div>
""", unsafe_allow_html=True)

if not os.path.exists(MODEL_PATH):
    st.error(
        "File model tidak ditemukan. Pastikan "
        "model_transfer_learning.keras tersedia di repository GitHub."
    )
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Model gagal dimuat: {e}")
    st.stop()

# =========================
# PAGE: CLASSIFICATION
# =========================
if page == "Klasifikasi Gambar":
    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.markdown("### 📤 Input Gambar")
        st.markdown(
            '<div class="small-muted">Unggah gambar JPG, JPEG, atau PNG.</div>',
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Pilih gambar",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Gambar yang diunggah", use_container_width=True)
            st.caption(f"Ukuran gambar asli: {image.width} × {image.height}")

            if st.button(
                "🔍 Klasifikasikan Gambar",
                type="primary",
                use_container_width=True
            ):
                img = image.resize((32, 32))
                img_array = np.asarray(img, dtype=np.float32)
                img_array = np.expand_dims(img_array, axis=0)

                with st.spinner("Model sedang menganalisis gambar..."):
                    probabilities = model.predict(
                        img_array, verbose=0
                    )[0]

                idx = int(np.argmax(probabilities))
                confidence = float(probabilities[idx])

                st.session_state.history.insert(0, {
                    "nama_file": uploaded_file.name,
                    "prediksi": CLASS_NAMES[idx],
                    "confidence": confidence * 100,
                    "probabilities": probabilities.tolist()
                })
                st.session_state.history = st.session_state.history[:20]

                st.session_state.last_result = {
                    "nama_file": uploaded_file.name,
                    "prediksi": CLASS_NAMES[idx],
                    "confidence": confidence * 100,
                    "probabilities": probabilities.tolist(),
                    "image": image.copy()
                }

    with right:
        st.markdown("### 🎯 Hasil Prediksi")

        result = st.session_state.get("last_result")

        if result:
            st.markdown(
                f"""
                <div class="panel">
                    <div class="small-muted">PREDIKSI UTAMA</div>
                    <h1 style="color:{accent}; margin-bottom:4px;">
                        {result['prediksi']}
                    </h1>
                    <p class="small-muted">
                        File: {result['nama_file']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            m1, m2 = st.columns(2)
            m1.metric("Confidence", f"{result['confidence']:.2f}%")
            m2.metric("Jumlah Kelas", "10")

            st.markdown("#### 📊 Probabilitas Setiap Kelas")

            chart_data = pd.DataFrame({
                "Kelas": CLASS_NAMES,
                "Probabilitas": np.array(
                    result["probabilities"]
                ) * 100
            }).sort_values("Probabilitas", ascending=False)

            chart = alt.Chart(chart_data).mark_bar(
                cornerRadiusEnd=5,
                color=accent
            ).encode(
                x=alt.X(
                    "Probabilitas:Q",
                    title="Probabilitas (%)",
                    scale=alt.Scale(domain=[0, 100])
                ),
                y=alt.Y(
                    "Kelas:N",
                    sort="-x",
                    title=None
                ),
                tooltip=[
                    alt.Tooltip("Kelas:N"),
                    alt.Tooltip(
                        "Probabilitas:Q",
                        format=".2f"
                    )
                ]
            ).properties(height=320).configure_view(
                strokeOpacity=0
            ).configure_axis(
                labelColor=text,
                titleColor=muted,
                gridColor=border
            )

            st.altair_chart(chart, use_container_width=True)

            st.caption(
                "Confidence merupakan probabilitas tertinggi dari model, "
                "bukan jaminan bahwa prediksi selalu benar."
            )
        else:
            st.markdown(
                """
                <div class="panel">
                    <h3>Belum ada hasil</h3>
                    <p class="small-muted">
                        Unggah gambar di sebelah kiri, lalu tekan tombol
                        klasifikasi untuk melihat hasil prediksi.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

# =========================
# PAGE: HISTORY
# =========================
elif page == "Riwayat Prediksi":
    st.markdown("### 🕘 Riwayat Prediksi")
    st.caption("Riwayat tersimpan selama sesi aplikasi berlangsung.")

    history = st.session_state.history

    if history:
        history_df = pd.DataFrame([
            {
                "Nama File": h["nama_file"],
                "Hasil Prediksi": h["prediksi"],
                "Confidence (%)": round(h["confidence"], 2)
            }
            for h in history
        ])
        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        if st.button("🗑️ Hapus Riwayat"):
            st.session_state.history = []
            st.session_state.pop("last_result", None)
            st.rerun()
    else:
        st.info("Belum ada riwayat. Coba klasifikasikan gambar terlebih dahulu.")

# =========================
# PAGE: ABOUT
# =========================
elif page == "Tentang Proyek":
    st.markdown("### ℹ️ Tentang Proyek")
    st.markdown(
        """
        Aplikasi ini merupakan demonstrasi klasifikasi gambar CIFAR-10
        menggunakan model deep learning dengan pendekatan Transfer Learning.

        **Teknologi yang digunakan**
        - Python dan Streamlit untuk aplikasi web.
        - TensorFlow/Keras untuk menjalankan model.
        - MobileNetV2 sebagai model dasar Transfer Learning.
        - Altair untuk visualisasi probabilitas prediksi.

        **Sepuluh kelas CIFAR-10**
        Pesawat, mobil, burung, kucing, rusa, anjing, katak, kuda, kapal, dan truk.

        Catatan: gambar dari luar dataset CIFAR-10 dapat memiliki karakteristik
        berbeda dari data pelatihan sehingga hasil prediksi bisa kurang akurat.
        """
    )

st.divider()
st.markdown(
    '<p class="small-muted" style="text-align:center;">'
    'NeuroVision AI • CNN Image Classifier • MobileNetV2'
    '</p>',
    unsafe_allow_html=True
)
