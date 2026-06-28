"""
PAGE 4: PERBANDINGAN KOMODITAS
- Multiselect filter (up to 5 commodities)
- Multi-line trend chart with normalization option
- Correlation matrix heatmap
- NO clustering/K-Means/PCA (focus on retail dynamics)
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_loader import (
    load_features_dataset,
    get_commodity_list,
    filter_by_date_range
)
from utils.visualization import (
    create_line_chart,
    create_heatmap,
    format_currency,
    COLORS
)

# Page config
st.set_page_config(page_title="Perbandingan - PANTAU PASAR", page_icon="🔄", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .comparison-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div style='text-align: center; padding: 1rem 0 2rem 0;'>
        <h1 style='color: #1E40AF; margin-bottom: 0.3rem;'>🔄 Perbandingan Komoditas</h1>
        <p style='font-size: 1rem; color: #475569;'>
            Analisis Komparatif & Korelasi Harga
        </p>
    </div>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_comparison_data():
    df = load_features_dataset()
    return df

df = load_comparison_data()
commodity_list = get_commodity_list(df)

# === SIDEBAR ===
st.sidebar.markdown("## 🔍 Filter Perbandingan")

# Commodity multiselect
selected_commodities = st.sidebar.multiselect(
    "Pilih Komoditas (Maks 5)",
    options=commodity_list,
    default=commodity_list[:3] if len(commodity_list) >= 3 else commodity_list,
    max_selections=5,
    help="Pilih 2-5 komoditas untuk perbandingan"
)

# Date range
min_date = df['Tanggal'].min().date()
max_date = df['Tanggal'].max().date()

date_range = st.sidebar.date_input(
    "Rentang Tanggal",
    value=(max_date - pd.Timedelta(days=180), max_date),
    min_value=min_date,
    max_value=max_date,
    help="Pilih rentang tanggal untuk analisis"
)

# Normalization toggle
normalize = st.sidebar.checkbox(
    "Normalisasi Skala (0-100)",
    value=False,
    help="Normalisasi harga ke skala 0-100 untuk perbandingan fair antar komoditas dengan range harga berbeda"
)

st.sidebar.markdown("---")
st.sidebar.info("""
    **💡 Tips:**
    - Pilih komoditas dengan range harga serupa untuk perbandingan langsung
    - Gunakan normalisasi untuk membandingkan komoditas volatile vs stabil
    - Korelasi tinggi (>0.7) menunjukkan pergerakan harga sejalan
""")

# Validation
if len(selected_commodities) < 2:
    st.warning("⚠️ Pilih minimal 2 komoditas untuk perbandingan.")
    st.stop()

# Filter data
filtered_df = df[df['Komoditas'].isin(selected_commodities)].copy()

if len(date_range) == 2:
    filtered_df = filter_by_date_range(filtered_df, date_range[0], date_range[1])

# === SECTION 1: MULTI-LINE TREND CHART ===
st.markdown("### 📈 Tren Harga Komparatif")

# Apply normalization if selected
if normalize:
    st.info("📊 **Normalisasi aktif:** Harga dinormalisasi ke skala 0-100 berdasarkan min-max per komoditas.")
    
    for commodity in selected_commodities:
        mask = filtered_df['Komoditas'] == commodity
        prices = filtered_df.loc[mask, 'Harga']
        
        if len(prices) > 0:
            min_price = prices.min()
            max_price = prices.max()
            
            if max_price > min_price:
                filtered_df.loc[mask, 'Harga_Normalized'] = ((prices - min_price) / (max_price - min_price)) * 100
            else:
                filtered_df.loc[mask, 'Harga_Normalized'] = 50  # Constant price
    
    y_col = 'Harga_Normalized'
    y_title = "Harga Ternormalisasi (0-100)"
else:
    y_col = 'Harga'
    y_title = "Harga (Rp)"

# Create line chart
if len(filtered_df) > 0:
    fig_trend = create_line_chart(
        filtered_df,
        x_col='Tanggal',
        y_col=y_col,
        color_col='Komoditas',
        title=f"Perbandingan Tren: {', '.join(selected_commodities)}",
        xaxis_title="Tanggal",
        yaxis_title=y_title,
        show_legend=True,
        height=500
    )
    st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False})
else:
    st.warning("Tidak ada data untuk rentang tanggal yang dipilih.")

st.markdown("---")

# === SECTION 2: STATISTICS COMPARISON ===
st.markdown("### 📊 Statistik Komparatif")

stats_data = []
for commodity in selected_commodities:
    comm_data = filtered_df[filtered_df['Komoditas'] == commodity]['Harga']
    
    if len(comm_data) > 0:
        stats_data.append({
            'Komoditas': commodity,
            'Mean': comm_data.mean(),
            'Median': comm_data.median(),
            'Std': comm_data.std(),
            'CV (%)': (comm_data.std() / comm_data.mean()) * 100,
            'Min': comm_data.min(),
            'Max': comm_data.max(),
            'Range': comm_data.max() - comm_data.min()
        })

if len(stats_data) > 0:
    stats_df = pd.DataFrame(stats_data)
    
    # Format and display
    styled_stats = stats_df.style.format({
        'Mean': lambda x: format_currency(x),
        'Median': lambda x: format_currency(x),
        'Std': lambda x: format_currency(x),
        'CV (%)': '{:.2f}%',
        'Min': lambda x: format_currency(x),
        'Max': lambda x: format_currency(x),
        'Range': lambda x: format_currency(x)
    }).background_gradient(subset=['CV (%)'], cmap='RdYlGn_r')
    
    st.dataframe(styled_stats, use_container_width=True, hide_index=True)
    
    # Insights
    most_volatile = stats_df.loc[stats_df['CV (%)'].idxmax()]
    least_volatile = stats_df.loc[stats_df['CV (%)'].idxmin()]
    highest_price = stats_df.loc[stats_df['Mean'].idxmax()]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div class='comparison-card'>
                <div style='font-size: 0.85rem; color: #64748B; margin-bottom: 0.3rem;'>
                    PALING VOLATILE
                </div>
                <div style='font-size: 1.3rem; font-weight: 600; color: #DC2626;'>
                    {most_volatile['Komoditas']}
                </div>
                <div style='font-size: 0.9rem; color: #475569; margin-top: 0.3rem;'>
                    CV: {most_volatile['CV (%)']:.2f}%
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class='comparison-card'>
                <div style='font-size: 0.85rem; color: #64748B; margin-bottom: 0.3rem;'>
                    PALING STABIL
                </div>
                <div style='font-size: 1.3rem; font-weight: 600; color: #16A34A;'>
                    {least_volatile['Komoditas']}
                </div>
                <div style='font-size: 0.9rem; color: #475569; margin-top: 0.3rem;'>
                    CV: {least_volatile['CV (%)']:.2f}%
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class='comparison-card'>
                <div style='font-size: 0.85rem; color: #64748B; margin-bottom: 0.3rem;'>
                    HARGA TERTINGGI
                </div>
                <div style='font-size: 1.3rem; font-weight: 600; color: #1E40AF;'>
                    {highest_price['Komoditas']}
                </div>
                <div style='font-size: 0.9rem; color: #475569; margin-top: 0.3rem;'>
                    {format_currency(highest_price['Mean'])}
                </div>
            </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# === SECTION 3: CORRELATION MATRIX ===
st.markdown("### 🔗 Matriks Korelasi Harga")

st.markdown("""
    <p style='color: #475569; margin-bottom: 1rem;'>
        Korelasi Pearson antar komoditas. Nilai mendekati <strong>+1</strong> menunjukkan pergerakan harga sejalan,
        nilai mendekati <strong>-1</strong> menunjukkan pergerakan berlawanan, nilai mendekati <strong>0</strong> menunjukkan tidak ada korelasi.
    </p>
""", unsafe_allow_html=True)

# Pivot data for correlation
if len(filtered_df) > 0:
    # Create pivot table
    pivot_df = filtered_df.pivot_table(
        index='Tanggal',
        columns='Komoditas',
        values='Harga',
        aggfunc='first'
    )
    
    # Calculate correlation
    if len(pivot_df.columns) >= 2:
        corr_matrix = pivot_df.corr()
        
        # Create heatmap
        fig_corr = create_heatmap(
            corr_matrix,
            title="Correlation Matrix: Price Co-movement",
            height=max(400, len(selected_commodities) * 80),
            colorscale='RdBu_r'
        )
        st.plotly_chart(fig_corr, use_container_width=True, config={'displayModeBar': False})
        
        # Interpretation guide
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div style='background-color: #DCFCE7; padding: 0.8rem; border-radius: 6px; text-align: center;'>
                    <strong style='color: #16A34A;'>Korelasi Kuat Positif</strong><br>
                    <span style='font-size: 0.85rem; color: #475569;'>0.7 ≤ r ≤ 1.0</span><br>
                    <span style='font-size: 0.8rem; color: #64748B;'>Harga bergerak sejalan</span>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div style='background-color: #FEF3C7; padding: 0.8rem; border-radius: 6px; text-align: center;'>
                    <strong style='color: #F59E0B;'>Korelasi Lemah</strong><br>
                    <span style='font-size: 0.85rem; color: #475569;'>-0.3 < r < 0.3</span><br>
                    <span style='font-size: 0.8rem; color: #64748B;'>Tidak ada hubungan jelas</span>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div style='background-color: #FEE2E2; padding: 0.8rem; border-radius: 6px; text-align: center;'>
                    <strong style='color: #DC2626;'>Korelasi Kuat Negatif</strong><br>
                    <span style='font-size: 0.85rem; color: #475569;'>-1.0 ≤ r ≤ -0.7</span><br>
                    <span style='font-size: 0.8rem; color: #64748B;'>Harga berlawanan arah</span>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Find strongest correlations
        # Get upper triangle (exclude diagonal)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'Komoditas 1': corr_matrix.columns[i],
                    'Komoditas 2': corr_matrix.columns[j],
                    'Korelasi': corr_matrix.iloc[i, j]
                })
        
        if len(corr_pairs) > 0:
            corr_pairs_df = pd.DataFrame(corr_pairs).sort_values('Korelasi', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔗 Top 3 Korelasi Tertinggi:**")
                top_corr = corr_pairs_df.head(3)
                for idx, row in top_corr.iterrows():
                    st.markdown(f"""
                        <div style='background-color: #F8FAFC; padding: 0.6rem; margin: 0.3rem 0; border-radius: 4px; border-left: 3px solid #16A34A;'>
                            <strong>{row['Komoditas 1']}</strong> ↔ <strong>{row['Komoditas 2']}</strong><br>
                            <span style='color: #16A34A; font-weight: 600;'>r = {row['Korelasi']:.3f}</span>
                        </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("**📉 Top 3 Korelasi Terendah:**")
                bottom_corr = corr_pairs_df.tail(3).sort_values('Korelasi')
                for idx, row in bottom_corr.iterrows():
                    color = '#DC2626' if row['Korelasi'] < 0 else '#64748B'
                    st.markdown(f"""
                        <div style='background-color: #F8FAFC; padding: 0.6rem; margin: 0.3rem 0; border-radius: 4px; border-left: 3px solid {color};'>
                            <strong>{row['Komoditas 1']}</strong> ↔ <strong>{row['Komoditas 2']}</strong><br>
                            <span style='color: {color}; font-weight: 600;'>r = {row['Korelasi']:.3f}</span>
                        </div>
                    """, unsafe_allow_html=True)
    else:
        st.warning("Minimal 2 komoditas diperlukan untuk analisis korelasi.")
else:
    st.warning("Tidak ada data untuk analisis korelasi.")

st.markdown("---")

# === INSIGHTS ===
st.markdown("### 💡 Interpretasi & Rekomendasi")

st.success("""
**✅ Untuk Trader & Policy Maker:**

1. **Komoditas dengan korelasi tinggi** (r > 0.7):
   - Cenderung mengalami spike/drop bersamaan
   - Intervensi harga pada satu komoditas bisa mempengaruhi komoditas berkorelasi
   - Contoh: Cabai rawit ↔ Cabai merah besar (keduanya supply sayuran)

2. **Komoditas dengan korelasi rendah** (|r| < 0.3):
   - Pergerakan harga independen
   - Diversifikasi stocking lebih aman
   - Substitusi tidak efektif

3. **Strategi Substitusi:**
   - Korelasi tinggi + kategori serupa → kandidat substitusi baik
   - Contoh: Beras medium ↔ Beras premium
""")

st.info("""
**📌 Catatan Metodologi:**

Dashboard ini **fokus pada retail dynamics Pasar Seketeng** tanpa clustering/K-Means/PCA kompleks.
Analisis korelasi sederhana lebih mudah diinterpretasi untuk decision-making praktis.

Untuk analisis lanjutan (spatial clustering, demand forecasting), lihat Rekomendasi Penelitian Lanjutan di halaman Metodologi.
""")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #64748B; font-size: 0.85rem;'>
        <p>Analisis Komparatif • PANTAU PASAR Dashboard</p>
    </div>
""", unsafe_allow_html=True)
