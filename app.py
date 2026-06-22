import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from skimage.feature import graycomatrix, graycoprops
import pickle
import streamlit.components.v1 as components
import json
import base64
import io

st.set_page_config(page_title="NILAM AI - Pro Deployment", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for Dark Premium Dashboard & Animated Background
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Font & Theme Styling */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #f1f5f9;
    }
    
    /* Animated Gradient Background */
    .stApp {
        background: linear-gradient(-45deg, #060913, #0f172a, #130f24, #064e3b);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        background-attachment: fixed;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Custom Webkit Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #0f172a; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #475569; }

    /* Hide standard Streamlit UI */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Premium Header Banner */
    .custom-header {
        background: rgba(15, 23, 42, 0.4);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-top: 1px solid rgba(255, 255, 255, 0.15);
        padding: 3rem 2rem;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 2.5rem;
        margin-top: 1rem;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.7);
        position: relative;
        overflow: hidden;
    }

    /* Floating Animation for Header Icon */
    @keyframes float {
        0% { transform: translateY(0px) scale(1); filter: drop-shadow(0 0 10px rgba(52, 211, 153, 0.4)); }
        50% { transform: translateY(-12px) scale(1.05); filter: drop-shadow(0 0 25px rgba(129, 140, 248, 0.8)); }
        100% { transform: translateY(0px) scale(1); filter: drop-shadow(0 0 10px rgba(52, 211, 153, 0.4)); }
    }
    
    .header-icon {
        animation: float 4s ease-in-out infinite;
        margin-bottom: 12px;
    }

    .custom-header h1 {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 3.8rem;
        letter-spacing: -1.5px;
        margin: 0;
        background: linear-gradient(135deg, #10b981 0%, #34d399 25%, #818cf8 75%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 10px 30px rgba(16, 185, 129, 0.2);
    }
    
    .custom-header p {
        color: #94a3b8;
        font-weight: 500;
        margin-top: 12px;
        font-size: 1.15rem;
        letter-spacing: 0.5px;
    }

    /* Glassmorphism Container Card */
    .glass-card {
        background: rgba(15, 23, 42, 0.35);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-top: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.4);
        margin-bottom: 20px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .glass-card:hover {
        border-color: rgba(16, 185, 129, 0.25);
        box-shadow: 0 15px 35px -10px rgba(16, 185, 129, 0.15);
    }

    .card-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 1.3rem;
        color: #f8fafc;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 10px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 12px;
    }

    /* Custom File Uploader Override */
    .stFileUploader > div > div {
        background-color: rgba(15, 23, 42, 0.5) !important;
        border: 2px dashed rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        padding: 2.5rem !important;
        transition: all 0.4s ease;
    }
    
    .stFileUploader > div > div:hover {
        border-color: #34d399 !important;
        background-color: rgba(16, 185, 129, 0.05) !important;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.1) inset;
    }

    /* Custom CSS Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(15, 23, 42, 0.3);
        padding: 8px 8px 0 8px;
        border-radius: 16px 16px 0 0;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: #94a3b8 !important;
        border: none !important;
        padding: 14px 28px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        border-radius: 8px 8px 0 0 !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #f8fafc !important;
        background: rgba(255, 255, 255, 0.03) !important;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #34d399 !important;
        border-bottom: 3px solid #34d399 !important;
        background: linear-gradient(0deg, rgba(16, 185, 129, 0.1) 0%, transparent 100%) !important;
    }

</style>
""", unsafe_allow_html=True)

# Render Header Banner
st.markdown("""
<div class="custom-header">
    <div class="header-content">
        <svg class="header-icon" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="url(#gradient)" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <defs>
                <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#34d399" />
                    <stop offset="100%" stop-color="#818cf8" />
                </linearGradient>
            </defs>
            <path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"></path>
        </svg>
        <h1>NILAM AI PRO</h1>
        <p>Sistem Klasifikasi Varian Daun Nilam Terstandarisasi berbasis Support Vector Machine</p>
    </div>
</div>
""", unsafe_allow_html=True)

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

def pil_to_base64(img_pil):
    buffered = io.BytesIO()
    img_pil.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

# Class mapping & descriptions
classes = {
    0: "Nilam Batik",
    1: "Nilam Biasa",
    2: "Nilam Seledri"
}

class_descriptions = {
    0: "Varian daun nilam dengan corak variegata yang menyerupai motif batik. Memiliki kandungan senyawa aktif patchouli alcohol yang unik dan ketahanan fisiologis tanaman terhadap beberapa patogen tertentu.",
    1: "Varietas standard (Sidikalang/Lhokseumawe) dengan daun berbentuk bulat telur, lebar, berwarna hijau segar, dan bergelombang lembut. Merupakan jenis komoditas utama petani karena menghasilkan Patchouli Oil dalam jumlah tinggi.",
    2: "Varian Tapak Tuan dengan struktur tepi daun yang bergerigi tajam menyerupai daun seledri. Tanaman ini memiliki karakteristik aroma patchouli yang kuat dan tajam dengan ketahanan fisik yang baik terhadap fluktuasi cuaca ekstrem."
}

feature_names = [
    'GLCM Contrast', 'GLCM Homogeneity', 'GLCM Energy', 'GLCM Correlation', 'GLCM Dissimilarity', 'GLCM ASM',
    'H Mean (Hue)', 'H Std (Hue)', 'S Mean (Sat)', 'S Std (Sat)', 'V Mean (Val)', 'V Std (Val)'
]

def extract_features(img_pil):
    gray_arr_raw = np.array(img_pil.convert('L'))
    leaf_mask = (gray_arr_raw < 210) & (gray_arr_raw > 20)
    if np.sum(leaf_mask) == 0:
        leaf_mask = np.ones_like(gray_arr_raw, dtype=bool)

    current_brightness = np.mean(gray_arr_raw[leaf_mask])
    factor = 115.0 / current_brightness
    factor = max(0.5, min(1.8, factor))

    enhancer = ImageEnhance.Brightness(img_pil)
    img_normalized = enhancer.enhance(factor)

    gray_arr = np.array(img_normalized.convert('L'))
    hsv_arr = np.array(img_normalized.convert('HSV'))

    h = hsv_arr[:, :, 0][leaf_mask]
    s = hsv_arr[:, :, 1][leaf_mask]
    v = hsv_arr[:, :, 2][leaf_mask]
    color_features = [np.mean(h), np.std(h), np.mean(s), np.std(s), np.mean(v), np.std(v)]

    glcm = graycomatrix(gray_arr, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    texture_features = [
        np.mean(graycoprops(glcm, prop)) for prop in ['contrast', 'homogeneity', 'energy', 'correlation', 'dissimilarity', 'ASM']
    ]

    combined_features = texture_features + color_features
    return np.array(combined_features).reshape(1, -1)

# Tab Navigation Setup
tab1, tab2, tab3 = st.tabs(["🩺 Diagnosis Klasifikasi", "📊 Dashboard Fitur 3D", "🧠 Arsitektur Model"])

with tab1:
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <div class="card-title">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>
                Unggah Citra Uji
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            img_pil = Image.open(uploaded_file).convert('RGB')
            # Custom component for image scanning animation
            img_b64 = pil_to_base64(img_pil)
            components.html(f"""
            <div style="position: relative; width: 100%; border-radius: 16px; overflow: hidden; box-shadow: 0 10px 25px rgba(0,0,0,0.5);">
                <img src="data:image/jpeg;base64,{img_b64}" style="width: 100%; display: block;" />
                <div id="scanner" style="position: absolute; top: -10px; left: 0; width: 100%; height: 4px; background: #34d399; box-shadow: 0 0 20px 8px rgba(52, 211, 153, 0.4); animation: scan 3s ease-in-out infinite;"></div>
            </div>
            <style>
                @keyframes scan {{
                    0% {{ top: -10px; opacity: 0; }}
                    10% {{ opacity: 1; }}
                    90% {{ opacity: 1; }}
                    100% {{ top: 100%; opacity: 0; }}
                }}
            </style>
            <script>
                // Stop scanner after 3 seconds (when prediction finishes)
                setTimeout(() => {{
                    document.getElementById('scanner').style.display = 'none';
                }}, 3000);
            </script>
            """, height=400)
            
    with col2:
        st.markdown("""
        <div class="glass-card" style="margin-bottom: 0px;">
            <div class="card-title">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
                Hasil Identifikasi AI
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if uploaded_file is not None:
            with st.spinner("Mengekstraksi fitur dan melakukan komputasi matriks GLCM..."):
                try:
                    features = extract_features(img_pil)
                    features_scaled = scaler.transform(features)
                    
                    prediction = model.predict(features_scaled)[0]
                    predicted_class = classes.get(prediction, "Tidak Dikenal")
                    predicted_desc = class_descriptions.get(prediction, "")
                    
                    if hasattr(model, 'predict_proba'):
                        probs = model.predict_proba(features_scaled)[0]
                        confidence = np.max(probs) * 100
                    else:
                        probs = [0.0, 0.0, 0.0]
                        probs[prediction] = 1.0
                        confidence = 100.0
                    
                    st.session_state['features_raw'] = features[0]
                    st.session_state['features_scaled'] = features_scaled[0]
                    st.session_state['pred_idx'] = prediction
                    
                    labels_js = json.dumps(list(classes.values()))
                    data_js = json.dumps([round(p * 100, 2) for p in probs])
                    
                    # Colors based on prediction
                    card_color = "#34d399" # Default Biasa
                    if prediction == 0: card_color = "#c084fc" # Batik
                    elif prediction == 2: card_color = "#fbbf24" # Seledri
                    
                    # MEGA JS COMPONENT for 3D Tilt Card, Typing Effect, Chart.js, and Confetti
                    components.html(f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <script src="https://cdnjs.cloudflare.com/ajax/libs/vanilla-tilt/1.7.0/vanilla-tilt.min.js"></script>
                        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                        <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js"></script>
                        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&family=Plus+Jakarta+Sans:wght@400;500;700&display=swap" rel="stylesheet">
                        <style>
                            body {{ margin: 0; padding: 0; background: transparent; font-family: 'Plus Jakarta Sans', sans-serif; color: white; display: flex; flex-direction: column; align-items: center; }}
                            .tilt-card {{
                                width: 100%; max-width: 500px;
                                background: linear-gradient(135deg, rgba(15, 23, 42, 0.8), rgba(30, 41, 59, 0.6));
                                border: 1px solid rgba(255, 255, 255, 0.1);
                                border-top: 1px solid rgba(255, 255, 255, 0.2);
                                border-radius: 20px;
                                padding: 30px 20px;
                                text-align: center;
                                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
                                transform-style: preserve-3d;
                                position: relative;
                                overflow: hidden;
                            }}
                            .tilt-card::before {{
                                content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
                                background: radial-gradient(circle, {card_color}22 0%, transparent 60%);
                                z-index: -1; pointer-events: none;
                            }}
                            .title {{
                                font-size: 0.9rem; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; color: {card_color};
                                transform: translateZ(30px); margin-bottom: 10px;
                            }}
                            .prediction {{
                                font-family: 'Outfit', sans-serif; font-size: 3.5rem; font-weight: 800; margin: 0;
                                background: linear-gradient(135deg, #fff, {card_color});
                                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                                transform: translateZ(50px);
                                text-shadow: 0 5px 25px {card_color}55;
                            }}
                            .confidence {{
                                display: inline-block; padding: 8px 20px; background: rgba(0,0,0,0.3); border-radius: 99px;
                                border: 1px solid {card_color}55; font-weight: 700; font-size: 1rem; margin-top: 15px;
                                transform: translateZ(40px); box-shadow: 0 0 15px {card_color}33;
                                animation: pulse 2s infinite;
                            }}
                            @keyframes pulse {{
                                0% {{ box-shadow: 0 0 0 0 {card_color}66; }}
                                70% {{ box-shadow: 0 0 0 10px rgba(0,0,0,0); }}
                                100% {{ box-shadow: 0 0 0 0 rgba(0,0,0,0); }}
                            }}
                            .description {{
                                margin-top: 25px; color: #cbd5e1; font-size: 0.95rem; line-height: 1.6;
                                transform: translateZ(20px); text-align: justify;
                            }}
                            .chart-box {{
                                margin-top: 30px; width: 100%; max-width: 280px; margin-inline: auto;
                                transform: translateZ(40px);
                            }}
                        </style>
                    </head>
                    <body>
                        <div class="tilt-card" data-tilt data-tilt-max="10" data-tilt-speed="400" data-tilt-glare data-tilt-max-glare="0.2">
                            <div class="title">Terdiagnosis</div>
                            <div class="prediction">{predicted_class}</div>
                            <div class="confidence">Kepercayaan: {confidence:.2f}%</div>
                            
                            <div class="chart-box">
                                <canvas id="probChart"></canvas>
                            </div>
                            
                            <div class="description" id="typing-text"></div>
                        </div>

                        <script>
                            VanillaTilt.init(document.querySelector(".tilt-card"));
                            
                            // Typewriter effect
                            const text = "{predicted_desc}";
                            let i = 0;
                            function typeWriter() {{
                                if (i < text.length) {{
                                    document.getElementById("typing-text").innerHTML += text.charAt(i);
                                    i++;
                                    setTimeout(typeWriter, 20);
                                }}
                            }}
                            setTimeout(typeWriter, 500);

                            // Chart.js Doughnut
                            const ctx = document.getElementById('probChart').getContext('2d');
                            new Chart(ctx, {{
                                type: 'doughnut',
                                data: {{
                                    labels: {labels_js},
                                    datasets: [{{
                                        data: {data_js},
                                        backgroundColor: ['rgba(168, 85, 247, 0.8)', 'rgba(52, 211, 153, 0.8)', 'rgba(251, 191, 36, 0.8)'],
                                        borderColor: ['rgba(168, 85, 247, 1)', 'rgba(52, 211, 153, 1)', 'rgba(251, 191, 36, 1)'],
                                        borderWidth: 2, hoverOffset: 10
                                    }}]
                                }},
                                options: {{
                                    responsive: true, cutout: '75%',
                                    plugins: {{
                                        legend: {{ position: 'bottom', labels: {{ color: '#cbd5e1', font: {{ family: "'Plus Jakarta Sans', sans-serif" }} }} }}
                                    }},
                                    animation: {{ animateScale: true, animateRotate: true }}
                                }}
                            }});

                            // Confetti
                            var duration = 2.5 * 1000;
                            var end = Date.now() + duration;
                            (function frame() {{
                                confetti({{ particleCount: 5, angle: 60, spread: 55, origin: {{ x: 0 }}, colors: ['#34d399', '#818cf8', '#f59e0b'] }});
                                confetti({{ particleCount: 5, angle: 120, spread: 55, origin: {{ x: 1 }}, colors: ['#34d399', '#818cf8', '#f59e0b'] }});
                                if (Date.now() < end) {{ requestAnimationFrame(frame); }}
                            }}());
                        </script>
                    </body>
                    </html>
                    """, height=650)
                except Exception as e:
                    st.error(f"Terjadi error pengolahan: {e}")
        else:
            st.markdown("""
            <div style="text-align: center; padding: 60px 20px; color: #64748b;">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom: 16px; opacity: 0.5;">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="17 8 12 3 7 8"></polyline>
                    <line x1="12" y1="3" x2="12" y2="15"></line>
                </svg>
                <p style="font-size:1.1rem; font-weight:500; margin-bottom: 4px;">Menunggu Unggahan Citra</p>
                <p style="font-size:0.9rem;">Silakan unggah citra daun nilam di sebelah kiri untuk memulai klasifikasi otomatis.</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    if 'features_raw' in st.session_state:
        raw_vals = st.session_state['features_raw']
        scaled_vals = st.session_state['features_scaled']
        
        labels_json = json.dumps(feature_names)
        data_json = json.dumps([round(float(v), 3) for v in scaled_vals])
        raw_json = json.dumps([round(float(v), 4) for v in raw_vals])
        
        # MASSIVE HTML Component for Grid of 3D Cards & Radar Chart
        components.html(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/vanilla-tilt/1.7.0/vanilla-tilt.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&family=Plus+Jakarta+Sans:wght@400;600&display=swap" rel="stylesheet">
            <style>
                body {{ margin: 0; padding: 20px; font-family: 'Plus Jakarta Sans', sans-serif; background: transparent; color: white; }}
                .radar-container {{ width: 100%; max-width: 650px; height: 450px; margin: 0 auto 40px auto; }}
                .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; }}
                .f-card {{
                    background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255,255,255,0.08); border-radius: 16px;
                    padding: 20px; transform-style: preserve-3d; box-shadow: 0 10px 20px rgba(0,0,0,0.3);
                }}
                .f-title {{ font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; font-weight: 600; transform: translateZ(20px); display: block; }}
                .f-val {{ font-family: 'Outfit', sans-serif; font-size: 1.8rem; font-weight: 700; color: #f8fafc; margin: 10px 0; transform: translateZ(40px); display: block; }}
                .f-z {{ font-size: 0.8rem; color: #cbd5e1; transform: translateZ(30px); display: block; }}
                .bar-bg {{ background: #0f172a; height: 6px; border-radius: 3px; overflow: hidden; margin-top: 10px; transform: translateZ(20px); }}
                .bar-fg {{ height: 100%; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div style="text-align:center; color:#94a3b8; margin-bottom: 20px;">Analisis Vektor Z-Score dibandingkan dengan Mean Populasi Training</div>
            <div class="radar-container"><canvas id="radarChart"></canvas></div>
            
            <h3 style="font-family: Outfit; color: white; margin-bottom: 20px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom:10px;">Detail Vektor 12-Dimensi (Hover untuk 3D Effect)</h3>
            <div class="grid" id="card-grid"></div>

            <script>
                // Radar Chart
                const ctx = document.getElementById('radarChart').getContext('2d');
                new Chart(ctx, {{
                    type: 'radar',
                    data: {{
                        labels: {labels_json},
                        datasets: [{{
                            label: 'Z-Score Daun Uji', data: {data_json},
                            backgroundColor: 'rgba(52, 211, 153, 0.2)', borderColor: 'rgba(52, 211, 153, 1)',
                            pointBackgroundColor: 'rgba(99, 102, 241, 1)', pointBorderColor: '#fff',
                            borderWidth: 2, pointRadius: 4
                        }}]
                    }},
                    options: {{
                        responsive: true, maintainAspectRatio: false,
                        scales: {{ r: {{ angleLines: {{ color: 'rgba(255,255,255,0.1)' }}, grid: {{ color: 'rgba(255,255,255,0.1)' }}, pointLabels: {{ color: '#cbd5e1', font: {{ size: 11 }} }}, ticks: {{ display: false, min: -4, max: 4 }} }} }},
                        plugins: {{ legend: {{ labels: {{ color: '#f8fafc' }} }} }}
                    }}
                }});

                // Generate Cards
                const features = {labels_json};
                const scaled = {data_json};
                const raw = {raw_json};
                const grid = document.getElementById("card-grid");

                features.forEach((f, i) => {{
                    let s = scaled[i];
                    let r = raw[i];
                    let pct = Math.max(0, Math.min(100, (s + 3) / 6 * 100));
                    let color = s > 0.8 ? 'linear-gradient(90deg, #818cf8, #a855f7)' : (s < -0.8 ? 'linear-gradient(90deg, #f87171, #f59e0b)' : 'linear-gradient(90deg, #34d399, #10b981)');
                    
                    grid.innerHTML += `
                        <div class="f-card" data-tilt data-tilt-max="15" data-tilt-speed="400" data-tilt-glare data-tilt-max-glare="0.3">
                            <span class="f-title">${{f}}</span>
                            <span class="f-val">${{r}}</span>
                            <span class="f-z">Z-Score: ${{s > 0 ? '+'+s : s}}</span>
                            <div class="bar-bg"><div class="bar-fg" style="width: ${{pct}}%; background: ${{color}};"></div></div>
                        </div>
                    `;
                }});
                VanillaTilt.init(document.querySelectorAll(".f-card"));
            </script>
        </body>
        </html>
        """, height=1000)
    else:
        st.info("Silakan unggah citra di tab 'Diagnosis' terlebih dahulu.")

with tab3:
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>
            Sistem Flow & Architecture
        </div>
    """, unsafe_allow_html=True)
    
    # Custom interactive flow diagram component using JS
    components.html("""
    <!DOCTYPE html>
    <html>
    <head>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@600&family=Plus+Jakarta+Sans:wght@400;500&display=swap" rel="stylesheet">
        <style>
            body { margin: 0; padding: 20px; font-family: 'Plus Jakarta Sans', sans-serif; color: white; display: flex; justify-content: space-between; }
            .box { background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(52, 211, 153, 0.4); padding: 20px; border-radius: 12px; width: 30%; position: relative; box-shadow: 0 0 15px rgba(52,211,153,0.1); transition: transform 0.3s, box-shadow 0.3s; }
            .box:hover { transform: translateY(-5px); box-shadow: 0 0 25px rgba(52,211,153,0.3); border-color: rgba(52,211,153,0.8); }
            .title { font-family: 'Outfit'; color: #34d399; font-size: 1.2rem; margin-bottom: 10px; }
            .arrow { display: flex; align-items: center; color: #818cf8; font-size: 2rem; font-weight: bold; animation: slide 2s infinite; }
            @keyframes slide { 0% { transform: translateX(-5px); opacity: 0.5; } 50% { transform: translateX(5px); opacity: 1; } 100% { transform: translateX(-5px); opacity: 0.5; } }
            ul { padding-left: 20px; font-size: 0.9rem; line-height: 1.6; color: #cbd5e1; }
        </style>
    </head>
    <body>
        <div class="box">
            <div class="title">1. Preprocessing</div>
            <ul>
                <li>Grayscale Conversion</li>
                <li>Dynamic Threshold Masking (20-210)</li>
                <li>Brightness Normalization (Target 115)</li>
            </ul>
        </div>
        <div class="arrow">→</div>
        <div class="box" style="border-color: rgba(129, 140, 248, 0.4);">
            <div class="title" style="color: #818cf8;">2. Feature Extraction</div>
            <ul>
                <li><b>HSV Color:</b> Mean, Std Dev (6 Dimensi)</li>
                <li><b>GLCM Texture:</b> Contrast, Homogeneity, Energy, Correlation, Dissimilarity, ASM (6 Dimensi)</li>
            </ul>
        </div>
        <div class="arrow">→</div>
        <div class="box" style="border-color: rgba(192, 132, 252, 0.4);">
            <div class="title" style="color: #c084fc;">3. AI Classification</div>
            <ul>
                <li>StandardScaler Normalization</li>
                <li><b>SVM RBF Kernel</b></li>
                <li>Probabilitas Distribusi Kelas</li>
            </ul>
        </div>
    </body>
    </html>
    """, height=250)
    
    st.markdown("</div>", unsafe_allow_html=True)
