import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from skimage.feature import graycomatrix, graycoprops
import pickle

st.set_page_config(page_title="Klasifikasi Daun Nilam", page_icon="🌿", layout="centered")

st.title("🌿 Klasifikasi Daun Nilam dengan SVM")
st.write("Sistem ini mengekstraksi fitur warna (HSV) dan tekstur (GLCM) dari gambar, lalu memprediksi jenis daun nilam menggunakan Support Vector Machine (SVM).")

@st.cache_resource
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
    # 1. Penyesuaian Kecerahan
    enhancer = ImageEnhance.Brightness(img_pil)
    img_bright = enhancer.enhance(1.2)
    
    # Konversi ke OpenCV format (NumPy Array) BGR
    image = np.array(img_bright)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Diperkecil menjadi 128x128
    image = cv2.resize(image, (128, 128))

    # 2. FITUR WARNA (HSV)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_mean, h_std = np.mean(hsv_image[:,:,0]), np.std(hsv_image[:,:,0])
    s_mean, s_std = np.mean(hsv_image[:,:,1]), np.std(hsv_image[:,:,1])
    v_mean, v_std = np.mean(hsv_image[:,:,2]), np.std(hsv_image[:,:,2])
    color_features = [h_mean, h_std, s_mean, s_std, v_mean, v_std]

    # 3. FITUR TEKSTUR (GLCM)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    asm = graycoprops(glcm, 'ASM')[0, 0]
    
    texture_features = [contrast, correlation, energy, homogeneity, dissimilarity, asm]

    # Gabung semua fitur (12 Fitur)
    combined_features = color_features + texture_features
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
                
                # Output
                predicted_class = classes.get(prediction, "Kelas Tidak Dikenal")
                
                st.success(f"Prediksi: **{predicted_class}**")
                
                with st.expander("Lihat Detail Fitur"):
                    st.write(f"**Standarisasi Fitur:**\n{features_scaled[0]}")
                    
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
