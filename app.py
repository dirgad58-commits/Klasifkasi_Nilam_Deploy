import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from skimage.feature import graycomatrix, graycoprops
import pickle

st.set_page_config(page_title="Klasifikasi Daun Nilam", page_icon="🌿", layout="centered", initial_sidebar_state="collapsed")

# Inject Custom CSS for Professional Premium UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Global Font & Background */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background-color: #f8fafc;
        background-image: radial-gradient(circle at top right, #dcfce7 0%, transparent 40%),
                          radial-gradient(circle at bottom left, #ecfdf5 0%, transparent 40%);
    }

    /* Hide Streamlit UI Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Beautiful Header Banner */
    .main-header {
        background: linear-gradient(135deg, #166534 0%, #14532d 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 15px 30px rgba(22, 101, 52, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: "";
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
        transform: rotate(30deg);
        pointer-events: none;
    }

    .main-header h1 {
        color: white !important;
        font-weight: 700;
        margin: 0;
        padding: 0;
        font-size: 2.8rem;
        letter-spacing: -0.5px;
    }
    .main-header p {
        font-weight: 300;
        margin-top: 10px;
        opacity: 0.9;
        font-size: 1.15rem;
    }
    
    /* Result Badge */
    .result-card {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        color: white;
        padding: 20px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(34, 197, 94, 0.25);
        animation: scaleUp 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
        transform: scale(0.9);
        opacity: 0;
    }
    
    .result-card h3 {
        margin: 0;
        font-size: 1rem;
        font-weight: 400;
        opacity: 0.9;
        color: white !important;
    }
    .result-card h2 {
        margin: 5px 0 0 0;
        font-size: 2.2rem;
        font-weight: 700;
        color: white !important;
    }

    /* Subheader Styling */
    h3 {
        color: #1e293b !important;
        font-weight: 600 !important;
    }

    @keyframes scaleUp {
        to { transform: scale(1); opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Render Header
st.markdown("""
<div class="main-header">
    <h1>🌿 Identifikasi Daun Nilam</h1>
    <p>Sistem Deteksi Varian Berbasis Machine Learning (SVM)</p>
</div>
""", unsafe_allow_html=True)
def load_models():
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/best_svm_nilam_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return scaler, model

try:
    scaler, model = load_models()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# Mapping Kelas
classes = {
    0: "Nilam Batik",
    1: "Nilam Biasa",
    2: "Nilam Seledri"
}

def extract_features(img_pil):
    # 1. Masking & Normalisasi Kecerahan
    gray_arr_raw = np.array(img_pil.convert('L'))
    
    # Masker daun untuk analisis warna murni
    leaf_mask = (gray_arr_raw < 210) & (gray_arr_raw > 20)
    if np.sum(leaf_mask) == 0:
        leaf_mask = np.ones_like(gray_arr_raw, dtype=bool)

    current_brightness = np.mean(gray_arr_raw[leaf_mask])
    factor = 115.0 / current_brightness
    factor = max(0.5, min(1.8, factor))

    enhancer = ImageEnhance.Brightness(img_pil)
    img_normalized = enhancer.enhance(factor)

    # Konversi untuk Warna dan Tekstur menggunakan PIL
    gray_arr = np.array(img_normalized.convert('L'))
    hsv_arr = np.array(img_normalized.convert('HSV'))

    # 2. FITUR WARNA (HSV) - Dihitung HANYA pada area daun secara internal
    h = hsv_arr[:, :, 0][leaf_mask]
    s = hsv_arr[:, :, 1][leaf_mask]
    v = hsv_arr[:, :, 2][leaf_mask]
    color_features = [np.mean(h), np.std(h), np.mean(s), np.std(s), np.mean(v), np.std(v)]

    # 3. FITUR TEKSTUR (GLCM) - Dihitung pada seluruh citra abu-abu terstandarisasi
    glcm = graycomatrix(gray_arr, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    
    texture_features = [
        np.mean(graycoprops(glcm, prop)) for prop in ['contrast', 'homogeneity', 'energy', 'correlation', 'dissimilarity', 'ASM']
    ]

    # 4. Gabung semua fitur dengan urutan yang sama saat training: (Tekstur + Warna)
    combined_features = texture_features + color_features
    return np.array(combined_features).reshape(1, -1)

uploaded_file = st.file_uploader("Silakan unggah gambar daun nilam (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gambar yang Diunggah")
        img_pil = Image.open(uploaded_file).convert('RGB')
        st.image(img_pil, use_container_width=True)
        
    with col2:
        st.subheader("Hasil Analisis")
        with st.spinner("Mengekstraksi fitur dan melakukan prediksi..."):
            try:
                # Proses Ekstraksi
                features = extract_features(img_pil)
                
                # Transformasi Scaler
                features_scaled = scaler.transform(features)
                
                # Prediksi SVM
                prediction = model.predict(features_scaled)[0]
                
                # Output Premium
                predicted_class = classes.get(prediction, "Kelas Tidak Dikenal")
                
                st.markdown(f"""
                <div class="result-card">
                    <h3>Hasil Prediksi</h3>
                    <h2>{predicted_class}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("") # Spacer
                with st.expander("Tampilkan Metrik Fitur Teknis"):
                    st.write(f"**Standarisasi Fitur:**\n{features_scaled[0]}")
                    
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
