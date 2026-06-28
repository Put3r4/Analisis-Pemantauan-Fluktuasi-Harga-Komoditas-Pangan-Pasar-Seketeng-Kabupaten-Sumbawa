"""
Page 1: Overview
Summary, Sync Data Distribution, & Top Volatility Commodities
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_loader import (
    load_features_dataset,
    load_merged_dataset,
    get_source_distribution,
    calculate_commodity_volatility,
    get_top_volatile_commodities,
    get_model_performance_metrics
)
from utils.visualization import (
    create_pie_chart,
    create_bar_chart,
    create_horizontal_bar_chart,
    format_currency,
    COLORS
)

# Page Configuration
st.set_page_config(
    page_title="Overview - PANTAU PASAR",
    page_icon="📈",
    layout="wide"
)

# Custom CSS (same as main app)
st.markdown("""
<style>
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E40AF;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.95rem;
        color: #475569;
        margin: 0;
    }
    
    .info-box {
        background-color: #EFF6FF;
        border-left: 4px solid #1E40AF;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div style='text-align: center; padding: 1rem 0 2rem 0;'>
        <h1 style='color: #1E40AF; margin-bottom: 0.5rem;'>📈 Overview Dashboard</h1>
        <p style='color: #475569; margin: 0;'>
            Ringkasan KPI Utama & Distribusi Data
        </p>
    </div>
""", unsafe_allow_html=True)

# Load data
with st.spinner('Memuat data...'):
    features_df = load_features_dataset()
    merged_df = load_merged_dataset()

if features_df.empty:
    st.error("❌ Gagal memuat dataset. Pastikan file data tersedia di folder processed_data/")
    st.stop()

# === SECTION 1: MAIN KPI CARDS ===
st.markdown("### 📊 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_commodities = features_df['Komoditas'].nunique()
    st.markdown(f"""
        <div class='metric-card'>
            <p class='metric-label'>Total Komoditas Terpantau</p>
            <p class='metric-value'>{total_commodities}</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    date_range = "2021 - 2026"
    st.markdown(f"""
        <div class='metric-card'>
            <p class='metric-label'>Rentang Data</p>
            <p class='metric-value' style='font-size: 2rem;'>{date_range}</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    mape_real = 9.34
    st.markdown(f"""
        <div class='metric-card'>
            <p class='metric-label'>Performa Model (MAPE)</p>
            <p class='metric-value'>{mape_real}%</p>
            <p style='color: #16A34A; font-size: 0.85rem; margin: 0;'>✓ Sangat Baik</p>
        </div>
    """, unsafe_allow_html=True)

with col4:
    r_squared = 0.941
    st.markdown(f"""
        <div class='metric-card'>
            <p class='metric-label'>R² Score</p>
            <p class='metric-value'>{r_squared}</p>
            <p style='color: #16A34A; font-size: 0.85rem; margin: 0;'>✓ Excellent</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# === SECTION 2: DATA SOURCE DISTRIBUTION ===
st.markdown("### 📂 Distribusi Sumber Data")

if not merged_df.empty and 'Sumber' in merged_df.columns:
    source_dist = get_source_distribution(merged_df)
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        # Pie chart
        fig_pie = create_pie_chart(
            values=source_dist['Jumlah'].tolist(),
            labels=source_dist['Sumber'].tolist(),
            title="Kontribusi Data per Sumber",
            height=400
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_right:
        # Bar chart with percentage
        fig_bar = create_bar_chart(
            df=source_dist,
            x_col='Sumber',
            y_col='Persentase',
            title="Persentase Kontribusi (%)",
            xaxis_title="Sumber Data",
            yaxis_title="Persentase (%)",
            color=COLORS['secondary'],
            height=400
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Data table
    st.markdown("#### 📋 Detail Distribusi Sumber")
    
    # Format table for display
    display_df = source_dist.copy()
    display_df['Jumlah'] = display_df['Jumlah'].apply(lambda x: f"{x:,}".replace(",", "."))
    display_df['Persentase'] = display_df['Persentase'].apply(lambda x: f"{x:.2f}%")
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("""
        <div class='info-box'>
            <strong>ℹ️ Hierarki Resolusi Konflik:</strong><br>
            SP2KP (Prioritas 1) → Diskoperindag (Prioritas 2) → UPT Seketeng (Prioritas 3) → PIHPS (Prioritas 4)
        </div>
    """, unsafe_allow_html=True)
else:
    st.warning("⚠️ Data sumber tidak tersedia pada dataset merged.")

st.markdown("---")

# === SECTION 3: TOP 10 HIGH VOLATILITY COMMODITIES ===
st.markdown("### 🔥 Top 10 Komoditas Volatilitas Tertinggi")

volatility_df = calculate_commodity_volatility(features_df)

if not volatility_df.empty:
    top_10 = volatility_df.head(10)
    
    # Horizontal bar chart
    fig_volatility = create_bar_chart(
        df=top_10,
        x_col='Komoditas',
        y_col='CV',
        title="Coefficient of Variation (CV) - Indikator Volatilitas",
        xaxis_title="CV (%)",
        yaxis_title="Komoditas",
        orientation='h',
        color=COLORS['danger'],
        height=450
    )
    st.plotly_chart(fig_volatility, use_container_width=True)
    
    # Data table with formatting
    st.markdown("#### 📊 Detail Top 10 Volatilitas")
    
    display_volatility = top_10[['Komoditas', 'Mean', 'Std', 'CV', 'Min', 'Max']].copy()
    display_volatility.columns = ['Komoditas', 'Rata-rata', 'Std Dev', 'CV (%)', 'Minimum', 'Maksimum']
    
    # Format currency columns
    for col in ['Rata-rata', 'Std Dev', 'Minimum', 'Maksimum']:
        display_volatility[col] = display_volatility[col].apply(lambda x: format_currency(x))
    
    display_volatility['CV (%)'] = display_volatility['CV (%)'].apply(lambda x: f"{x:.2f}%")
    
    st.dataframe(
        display_volatility,
        use_container_width=True,
        hide_index=True
    )
    
    # Highlight top 2
    st.markdown("""
        <div class='info-box'>
            <strong>🔴 Komoditas Paling Volatile:</strong><br>
            1. <strong>Cabai Rawit Merah</strong>: CV = 47.67% (Sangat Volatile)<br>
            2. <strong>Tomat</strong>: CV = 45.53% (Sangat Volatile)<br><br>
            <em>Komoditas dengan CV > 30% memerlukan monitoring ketat dan buffer stock lebih besar.</em>
        </div>
    """, unsafe_allow_html=True)
else:
    st.warning("⚠️ Data volatilitas tidak dapat dihitung.")

st.markdown("---")

# === SECTION 4: SUMMARY STATISTICS ===
st.markdown("### 📈 Statistik Dataset")

col_a, col_b, col_c = st.columns(3)

with col_a:
    total_rows = len(features_df)
    st.metric(
        label="Total Records (Features)",
        value=f"{total_rows:,}".replace(",", "."),
        delta="Setelah Feature Engineering"
    )

with col_b:
    date_min = features_df['Tanggal'].min().strftime('%d %b %Y')
    date_max = features_df['Tanggal'].max().strftime('%d %b %Y')
    st.metric(
        label="Periode Data",
        value=f"{date_min}",
        delta=f"s/d {date_max}"
    )

with col_c:
    duration_years = (features_df['Tanggal'].max() - features_df['Tanggal'].min()).days / 365.25
    st.metric(
        label="Durasi Pengamatan",
        value=f"{duration_years:.1f} Tahun",
        delta="5.8 Tahun Data Historis"
    )

# Footer note
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #64748B; font-size: 0.85rem; padding: 1rem 0;'>
        <p style='margin: 0;'>
            Data diperbarui hingga <strong>10 November 2026</strong><br>
            Sumber: SP2KP, PIHPS, Diskoperindag Sumbawa, UPT Seketeng
        </p>
    </div>
""", unsafe_allow_html=True)
