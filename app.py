import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from skimage.feature import graycomatrix, graycoprops
import pickle

st.set_page_config(page_title="Klasifikasi Daun Nilam", page_icon="🌿", layout="wide", initial_sidebar_state="collapsed")

# Inject Custom CSS for Ultra Premium Dark UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Font & Dark Theme Background */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background-color: #0b0f19;
        background-image: 
            radial-gradient(circle at 15% 50%, rgba(16, 185, 129, 0.08), transparent 25%),
            radial-gradient(circle at 85% 30%, rgba(5, 150, 105, 0.08), transparent 25%);
        color: #e2e8f0;
    }

    /* Hide Streamlit UI Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Beautiful Header Banner */
    .main-header {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.7) 100%);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 3rem 2rem;
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        text-align: center;
        margin-bottom: 3rem;
        margin-top: 1rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::after {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 2px;
        background: linear-gradient(90deg, transparent, #10b981, transparent);
    }

    .main-header h1 {
        font-weight: 800;
        margin: 0;
        padding: 0;
        font-size: 3.2rem;
        letter-spacing: -1px;
        background: -webkit-linear-gradient(45deg, #34d399, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .main-header p {
        color: #94a3b8;
        font-weight: 400;
        margin-top: 12px;
        font-size: 1.2rem;
        letter-spacing: 0.5px;
    }
    
    /* Result Badge */
    .result-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.15) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 40px 20px;
        border-radius: 24px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(16, 185, 129, 0.15), inset 0 0 20px rgba(16, 185, 129, 0.05);
        animation: scaleUp 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
        transform: scale(0.9);
        opacity: 0;
        backdrop-filter: blur(10px);
    }
    
    .result-card h3 {
        margin: 0;
        font-size: 1.1rem;
        font-weight: 600;
        color: #a7f3d0 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .result-card h2 {
        margin: 10px 0 0 0;
        font-size: 3rem;
        font-weight: 800;
        color: #fff !important;
        text-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
    }

    /* Override standard components */
    .stFileUploader > div > div {
        background-color: rgba(30, 41, 59, 0.4) !important;
        border: 2px dashed rgba(148, 163, 184, 0.2) !important;
        border-radius: 20px !important;
        padding: 2rem !important;
        transition: all 0.3s ease;
    }
    .stFileUploader > div > div:hover {
        border-color: #10b981 !important;
        background-color: rgba(30, 41, 59, 0.7) !important;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.1);
    }
    
    .stMarkdown h3 {
        color: #f8fafc !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }

    /* Metric code block */
    code {
        color: #34d399 !important;
        background: rgba(15, 23, 42, 0.8) !important;
        padding: 15px !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.05);
        display: block;
        font-size: 0.9rem !important;
    }

    @keyframes scaleUp {
        to { transform: scale(1); opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Render Header
st.markdown("""
<div class="main-header">
    <h1><span style="font-size:3rem;">🌿</span> Identifikasi Daun Nilam AI</h1>
    <p>AgriTech Dashboard &bull; SVM Classification Engine</p>
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
    
