import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from skimage.feature import graycomatrix, graycoprops
import pickle

st.set_page_config(page_title="Klasifikasi Daun Nilam", layout="wide", initial_sidebar_state="collapsed")

# Inject Custom CSS for Clean Enterprise UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Font & Clean Light Theme */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #fafafa;
        color: #171717;
    }
    
    .stApp {
        background-color: #fafafa;
    }

    /* Hide Streamlit UI Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Minimalist Header Banner */
    .main-header {
        background-color: #ffffff;
        padding: 3rem 2rem;
        border-radius: 12px;
        border: 1px solid #e5e5e5;
        text-align: center;
        margin-bottom: 2.5rem;
        margin-top: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02);
    }

    .main-header h1 {
        font-weight: 700;
        margin: 0;
        padding: 0;
        font-size: 2.5rem;
        color: #171717;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: #525252;
        font-weight: 400;
        margin-top: 8px;
        font-size: 1.1rem;
    }
    
    /* Result Badge */
    .result-card {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        padding: 30px 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(22, 163, 74, 0.05);
        animation: fadeIn 0.4s ease-out forwards;
    }
    
    .result-card h3 {
        margin: 0;
        font-size: 0.95rem;
        font-weight: 600;
        color: #166534 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .result-card h2 {
        margin: 8px 0 0 0;
        font-size: 2.5rem;
        font-weight: 700;
        color: #15803d !important;
    }

    /* Override standard components */
    .stFileUploader > div > div {
        background-color: #ffffff !important;
        border: 1px dashed #d4d4d8 !important;
        border-radius: 8px !important;
        padding: 2rem !important;
        transition: all 0.2s ease;
    }
    .stFileUploader > div > div:hover {
        border-color: #16a34a !important;
        background-color: #f8fafc !important;
    }
    
    .stMarkdown h3 {
        color: #171717 !important;
        font-weight: 600 !important;
        letter-spacing: -0.3px;
        border-bottom: 1px solid #e5e5e5;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }

    /* Metric code block */
    code {
        color: #0f172a !important;
        background: #f1f5f9 !important;
        padding: 12px !important;
        border-radius: 6px !important;
        border: 1px solid #e2e8f0;
        display: block;
        font-size: 0.85rem !important;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(5px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Render Header
st.markdown("""
<div class="main-header">
    <h1>Sistem Identifikasi Varian Daun Nilam</h1>
    <p>Model Klasifikasi Support Vector Machine</p>
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

# Tambahan CSS untuk Progress Bar
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #10b981;
    }
</style>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Prediksi", "Analisis Fitur", "Info Model"])

with tab1:
    uploaded_file = st.file_uploader("Silakan unggah gambar daun nilam (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.markdown('<h3>Citra Uji Masukan</h3>', unsafe_allow_html=True)
            img_pil = Image.open(uploaded_file).convert('RGB')
            st.image(img_pil, use_container_width=True)
            
        with col2:
            st.markdown('<h3>Hasil Analisis AI</h3>', unsafe_allow_html=True)
            with st.spinner("Mengekstraksi fitur dan melakukan prediksi..."):
                try:
                    # Proses Ekstraksi
                    features = extract_features(img_pil)
                    
                    # Transformasi Scaler
                    features_scaled = scaler.transform(features)
                    
                    # Prediksi SVM
                    prediction = model.predict(features_scaled)[0]
                    predicted_class = classes.get(prediction, "Kelas Tidak Dikenal")
                    
                    # Probabilitas (Confidence)
                    if hasattr(model, 'predict_proba'):
                        probs = model.predict_proba(features_scaled)[0]
                        confidence = np.max(probs) * 100
                    else:
                        confidence = 0.0
                    
                    # Output Premium
                    st.markdown(f"""
                    <div class="result-card">
                        <h3>Hasil Prediksi</h3>
                        <h2>{predicted_class}</h2>
                        <p style="margin-top:10px; color:#15803d; font-size:1.1rem; font-weight:600;">Tingkat Kepercayaan: {confidence:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write(" ")
                    st.markdown('<h3>Detail Probabilitas Kelas</h3>', unsafe_allow_html=True)
                    if hasattr(model, 'predict_proba'):
                        for idx, cls_name in classes.items():
                            st.write(f"{cls_name} ({probs[idx]*100:.1f}%)")
                            st.progress(float(probs[idx]))
                        
                    # Simpan fitur ke session state untuk Tab 2
                    st.session_state['features_raw'] = features[0]
                    st.session_state['features_scaled'] = features_scaled[0]
                        
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses gambar: {e}")

with tab2:
    st.markdown('<h3>Visualisasi Nilai Ekstraksi Fitur</h3>', unsafe_allow_html=True)
    if 'features_raw' in st.session_state:
        feature_names = ['Contrast', 'Homogeneity', 'Energy', 'Correlation', 'Dissimilarity', 'ASM', 
                         'H Mean', 'H Std', 'S Mean', 'S Std', 'V Mean', 'V Std']
        
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.write("**Nilai Fitur Tekstur (GLCM)**")
            tex_dict = dict(zip(feature_names[:6], st.session_state['features_raw'][:6]))
            st.bar_chart(tex_dict, height=250)
            
        with col_t2:
            st.write("**Nilai Fitur Warna (HSV)**")
            col_dict = dict(zip(feature_names[6:], st.session_state['features_raw'][6:]))
            st.bar_chart(col_dict, height=250)
            
        st.write("**Vektor Standardisasi (Input SVM):**")
        st.code(st.session_state['features_scaled'])
    else:
        st.info("Silakan unggah gambar di tab 'Prediksi' terlebih dahulu untuk melihat analisis fitur.")

with tab3:
    st.markdown('<h3>Tentang Model</h3>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric(label="Algoritma", value="Support Vector Machine")
    c2.metric(label="Akurasi Validasi", value="95.0%")
    c3.metric(label="Jumlah Fitur", value="12 Fitur Kombinasi")
    
    st.markdown("""
    **Pipeline Ekstraksi Data:**
    1. **Isolasi Objek:** Memisahkan latar belakang dari daun untuk ekstraksi warna murni.
    2. **Normalisasi Kecerahan:** Mengkalibrasi ulang cahaya agar citra terstandarisasi.
    3. **Ekstraksi HSV:** Menghitung nilai rata-rata (*Mean*) dan standar deviasi dari saluran *Hue*, *Saturation*, *Value*.
    4. **Ekstraksi GLCM:** Memindai pola tekstur daun di 4 sudut arah (0°, 45°, 90°, 135°) untuk mencari nilai *Contrast, Homogeneity, Energy, Correlation, Dissimilarity, ASM*.
    5. **Klasifikasi:** 12 atribut diselaraskan menggunakan *StandardScaler* dan diprediksi dengan model SVM ber-kernel RBF.
    """)
