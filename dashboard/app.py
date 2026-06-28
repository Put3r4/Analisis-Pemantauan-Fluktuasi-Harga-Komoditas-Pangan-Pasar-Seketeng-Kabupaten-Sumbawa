"""
PANTAU PASAR - Sistem Pendukung Keputusan Harga Pangan
Pasar Seketeng, Kabupaten Sumbawa

Main Entry Point & Navigation
"""

import streamlit as st
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="PANTAU PASAR - Kabupaten Sumbawa",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Light, Minimalist, Professional Design
st.markdown("""
<style>
    /* Main Theme Colors */
    :root {
        --primary-color: #1E40AF;
        --secondary-color: #0D9488;
        --bg-white: #FFFFFF;
        --bg-container: #F8FAFC;
        --border-color: #E2E8F0;
        --text-primary: #0F172A;
        --text-secondary: #475569;
        --success-color: #16A34A;
        --danger-color: #DC2626;
    }
    
    /* Remove default Streamlit padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Card Style */
    .metric-card {
        background-color: var(--bg-container);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 0.5rem 0;
    }
    
    /* Typography */
    h1, h2, h3 {
        color: var(--text-primary);
        font-weight: 600;
    }
    
    p, div {
        color: var(--text-secondary);
    }
    
    /* Buttons */
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
    }
    
    .stButton>button:hover {
        background-color: #1E3A8A;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: var(--bg-white);
    }
    
    /* Remove default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: var(--primary-color);
        font-weight: 600;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary);
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div style='text-align: center; padding: 2rem 0 3rem 0;'>
        <h1 style='color: #1E40AF; margin-bottom: 0.5rem;'>🏪 PANTAU PASAR</h1>
        <p style='font-size: 1.2rem; color: #475569; margin: 0;'>
            Sistem Pendukung Keputusan Harga Pangan
        </p>
        <p style='font-size: 0.95rem; color: #64748B; margin-top: 0.3rem;'>
            Pasar Seketeng, Kabupaten Sumbawa
        </p>
    </div>
""", unsafe_allow_html=True)

# Navigation Info
st.markdown("""
    <div style='background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; 
                padding: 1.5rem; margin: 2rem 0;'>
        <h3 style='margin-top: 0; color: #1E40AF;'>📊 Selamat Datang</h3>
        <p style='color: #475569; line-height: 1.6;'>
            Dashboard ini menyediakan analisis komprehensif dan prediksi harga pangan di Pasar Seketeng
            menggunakan teknologi Machine Learning (Random Forest Regressor).
        </p>
        <p style='color: #475569; margin-bottom: 0;'>
            <strong>Gunakan sidebar di sebelah kiri untuk navigasi antar halaman:</strong>
        </p>
    </div>
""", unsafe_allow_html=True)

# Page Overview
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class='metric-card'>
            <h4 style='color: #1E40AF; margin-top: 0;'>📈 1. Overview</h4>
            <p style='color: #475569; margin-bottom: 0;'>
                Ringkasan KPI utama, distribusi sumber data, dan komoditas volatilitas tertinggi.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class='metric-card'>
            <h4 style='color: #1E40AF; margin-top: 0;'>📊 2. Analisis EDA</h4>
            <p style='color: #475569; margin-bottom: 0;'>
                Eksplorasi data historis, tren temporal, dan pola missing values.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class='metric-card'>
            <h4 style='color: #1E40AF; margin-top: 0;'>🎯 3. Prediksi Harga</h4>
            <p style='color: #475569; margin-bottom: 0;'>
                Inferensi model ML, metrik evaluasi dual-layer, dan feature importance.
            </p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class='metric-card'>
            <h4 style='color: #0D9488; margin-top: 0;'>🔄 4. Perbandingan</h4>
            <p style='color: #475569; margin-bottom: 0;'>
                Analisis komparatif antar komoditas dan korelasi harga.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class='metric-card'>
            <h4 style='color: #0D9488; margin-top: 0;'>📚 5. Metodologi</h4>
            <p style='color: #475569; margin-bottom: 0;'>
                Dokumentasi akademis pipeline data, model, dan hyperparameter.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class='metric-card'>
            <h4 style='color: #64748B; margin-top: 0;'>ℹ️ Info Sistem</h4>
            <p style='color: #475569; margin-bottom: 0; font-size: 0.9rem;'>
                <strong>Periode Data:</strong> 4 Jan 2021 - 10 Nov 2026 (5.8 tahun)<br>
                <strong>Total Komoditas:</strong> 57 jenis bahan pokok<br>
                <strong>Model Accuracy:</strong> MAPE 9.34% (Sangat Baik)
            </p>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #64748B; font-size: 0.85rem; padding: 1rem 0;'>
        <p style='margin: 0;'>
            Dikembangkan untuk Skripsi Sistem Informasi Manajemen<br>
            Kabupaten Sumbawa • 2026
        </p>
    </div>
""", unsafe_allow_html=True)
