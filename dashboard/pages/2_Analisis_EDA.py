"""
PAGE 2: ANALISIS EDA (Exploratory Data Analysis)
- Sidebar Filters (Commodity, Date Range, Data Source)
- Tab 1: Statistik Deskriptif
- Tab 2: Distribusi & Temporal Trend
- Tab 3: Missing Value Transparency
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_loader import (
    load_features_dataset,
    load_merged_dataset,
    get_commodity_list,
    get_commodity_statistics,
    filter_by_date_range,
    filter_by_commodity
)
from utils.visualization import (
    create_line_chart,
    create_box_plot,
    create_pie_chart,
    create_multi_line_chart,
    format_currency,
    COLORS
)

# Page config
st.set_page_config(page_title="Analisis EDA - PANTAU PASAR", page_icon="📊", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stat-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E40AF;
        margin: 0.3rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div style='text-align: center; padding: 1rem 0 2rem 0;'>
        <h1 style='color: #1E40AF; margin-bottom: 0.3rem;'>📊 Analisis EDA</h1>
        <p style='font-size: 1rem; color: #475569;'>
            Exploratory Data Analysis - Tren, Distribusi, dan Pola Data
        </p>
    </div>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_eda_data():
    df = load_features_dataset()
    df_merged = load_merged_dataset()
    return df, df_merged

df, df_merged = load_eda_data()
commodity_list = get_commodity_list(df)

# === SIDEBAR FILTERS ===
st.sidebar.markdown("## 🔍 Filter Data")

# Commodity filter
selected_commodities = st.sidebar.multiselect(
    "Pilih Komoditas",
    options=commodity_list,
    default=[commodity_list[0]] if len(commodity_list) > 0 else [],
    help="Pilih satu atau lebih komoditas untuk dianalisis"
)

# Date range filter
min_date = df['Tanggal'].min().date()
max_date = df['Tanggal'].max().date()

date_range = st.sidebar.date_input(
    "Rentang Tanggal",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    help="Pilih rentang tanggal untuk analisis"
)

# Data source filter (if available)
if 'Sumber' in df.columns and df['Sumber'].notna().any():
    available_sources = df['Sumber'].dropna().unique().tolist()
    selected_sources = st.sidebar.multiselect(
        "Sumber Data",
        options=available_sources,
        default=available_sources,
        help="Filter berdasarkan sumber data"
    )
else:
    selected_sources = []

st.sidebar.markdown("---")
st.sidebar.info("""
    **Tips:**
    - Pilih 1-3 komoditas untuk perbandingan optimal
    - Gunakan rentang tanggal untuk fokus pada periode tertentu
    - Perhatikan pola seasonal dan spike events
""")

# Apply filters
if len(selected_commodities) == 0:
    st.warning("⚠️ Pilih minimal 1 komoditas untuk melihat analisis.")
    st.stop()

filtered_df = filter_by_commodity(df, selected_commodities)

if len(date_range) == 2:
    filtered_df = filter_by_date_range(filtered_df, date_range[0], date_range[1])

# === TABS ===
tab1, tab2, tab3 = st.tabs(["📈 Statistik Deskriptif", "📉 Distribusi & Trend", "🔍 Missing Values"])

# === TAB 1: STATISTIK DESKRIPTIF ===
with tab1:
    st.markdown("### 📈 Statistik Deskriptif per Komoditas")
    
    # Calculate statistics for each selected commodity
    stats_data = []
    for commodity in selected_commodities:
        stats = get_commodity_statistics(df, commodity)
        if stats:
            stats_data.append({
                'Komoditas': commodity,
                'Mean (Rp)': stats['mean'],
                'Median (Rp)': stats['median'],
                'Std (Rp)': stats['std'],
                'CV (%)': stats['cv'],
                'Min (Rp)': stats['min'],
                'Max (Rp)': stats['max'],
                'Count': stats['count']
            })
    
    if len(stats_data) > 0:
        stats_df = pd.DataFrame(stats_data)
        
        # Format and display
        styled_stats = stats_df.style.format({
            'Mean (Rp)': lambda x: format_currency(x),
            'Median (Rp)': lambda x: format_currency(x),
            'Std (Rp)': lambda x: format_currency(x),
            'CV (%)': '{:.2f}%',
            'Min (Rp)': lambda x: format_currency(x),
            'Max (Rp)': lambda x: format_currency(x),
            'Count': '{:,.0f}'
        }).background_gradient(subset=['CV (%)'], cmap='YlOrRd')
        
        st.dataframe(styled_stats, use_container_width=True, hide_index=True)
        
        # Key insights
        col1, col2, col3 = st.columns(3)
        
        highest_mean = stats_df.loc[stats_df['Mean (Rp)'].idxmax()]
        highest_cv = stats_df.loc[stats_df['CV (%)'].idxmax()]
        lowest_cv = stats_df.loc[stats_df['CV (%)'].idxmin()]
        
        with col1:
            st.markdown(f"""
                <div class='stat-box'>
                    <div class='stat-label'>Harga Tertinggi</div>
                    <div class='stat-value'>{format_currency(highest_mean['Mean (Rp)'])}</div>
                    <div style='color: #64748B; font-size: 0.85rem;'>{highest_mean['Komoditas']}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class='stat-box'>
                    <div class='stat-label'>Paling Volatile</div>
                    <div class='stat-value' style='color: #DC2626;'>{highest_cv['CV (%)']:.2f}%</div>
                    <div style='color: #64748B; font-size: 0.85rem;'>{highest_cv['Komoditas']}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class='stat-box'>
                    <div class='stat-label'>Paling Stabil</div>
                    <div class='stat-value' style='color: #16A34A;'>{lowest_cv['CV (%)']:.2f}%</div>
                    <div style='color: #64748B; font-size: 0.85rem;'>{lowest_cv['Komoditas']}</div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Tidak ada data statistik untuk komoditas yang dipilih.")

# === TAB 2: DISTRIBUSI & TEMPORAL TREND ===
with tab2:
    st.markdown("### 📉 Tren Temporal Harga")
    
    if len(filtered_df) > 0:
        # Line chart with moving averages
        col1, col2 = st.columns([3, 1])
        
        with col2:
            show_ma = st.checkbox("Tampilkan Moving Average", value=True)
            ma_window_7 = st.checkbox("MA 7 Hari", value=True, disabled=not show_ma)
            ma_window_14 = st.checkbox("MA 14 Hari", value=True, disabled=not show_ma)
        
        with col1:
            # Prepare data for multi-line chart
            if len(selected_commodities) == 1:
                commodity = selected_commodities[0]
                comm_data = filtered_df[filtered_df['Komoditas'] == commodity].copy()
                comm_data = comm_data.sort_values('Tanggal')
                
                y_columns = ['Harga']
                labels = ['Harga Aktual']
                
                if show_ma:
                    if ma_window_7 and 'Rolling_Mean_7' in comm_data.columns:
                        y_columns.append('Rolling_Mean_7')
                        labels.append('MA 7 Hari')
                    if ma_window_14 and 'Rolling_Mean_14' in comm_data.columns:
                        y_columns.append('Rolling_Mean_14')
                        labels.append('MA 14 Hari')
                
                fig = create_multi_line_chart(
                    comm_data,
                    x_col='Tanggal',
                    y_cols=y_columns,
                    labels=labels,
                    title=f"Tren Harga: {commodity}",
                    xaxis_title="Tanggal",
                    yaxis_title="Harga (Rp)",
                    height=450
                )
            else:
                # Multiple commodities
                fig = create_line_chart(
                    filtered_df,
                    x_col='Tanggal',
                    y_col='Harga',
                    color_col='Komoditas',
                    title="Perbandingan Tren Harga Multi-Komoditas",
                    xaxis_title="Tanggal",
                    yaxis_title="Harga (Rp)",
                    height=450
                )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("---")
        
        # Box plot for distribution
        st.markdown("### 📊 Distribusi Harga")
        
        fig_box = create_box_plot(
            filtered_df,
            x_col='Komoditas',
            y_col='Harga',
            title="Distribusi Harga per Komoditas (Box Plot)",
            xaxis_title="Komoditas",
            yaxis_title="Harga (Rp)",
            height=400
        )
        st.plotly_chart(fig_box, use_container_width=True, config={'displayModeBar': False})
        
        st.info("""
            **Interpretasi Box Plot:**
            - **Kotak**: Rentang antar kuartil (Q1 - Q3), berisi 50% data tengah
            - **Garis tengah**: Median (Q2)
            - **Whiskers**: Rentang data normal (1.5 × IQR)
            - **Titik**: Outliers (harga anomali)
        """)
    else:
        st.warning("Tidak ada data untuk rentang tanggal yang dipilih.")

# === TAB 3: MISSING VALUES TRANSPARENCY ===
with tab3:
    st.markdown("### 🔍 Transparansi Missing Values")
    
    st.markdown("""
        <p style='color: #475569; line-height: 1.8;'>
            Dataset harga pangan memiliki karakteristik <strong>missing values alami</strong> karena:
        </p>
        <ul style='color: #475569; line-height: 1.8;'>
            <li>Pasar tradisional <strong>tutup di hari weekend</strong> dan hari libur nasional</li>
            <li>Tidak semua komoditas diperdagangkan setiap hari</li>
            <li>Beberapa sumber data memiliki <strong>frekuensi update berbeda</strong></li>
        </ul>
    """, unsafe_allow_html=True)
    
    # Visualize missing data pattern
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Pipeline Data ETL")
        
        # Data volume flow
        pipeline_data = {
            'Tahap': [
                'Raw (4 sumber)',
                'Merged',
                'Preprocessed (Grid)',
                'Feature Engineered'
            ],
            'Jumlah Baris': [53949, 39601, 121980, 76363],
            'Keterangan': [
                'Data mentah',
                'Deduplikasi -26.6%',
                'Calendar grid expansion',
                'Setelah dropna'
            ]
        }
        pipeline_df = pd.DataFrame(pipeline_data)
        
        st.dataframe(
            pipeline_df.style.format({'Jumlah Baris': '{:,.0f}'}),
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.markdown("#### Missing Value Handling")
        
        # Pie chart: Actual vs Imputed
        missing_data = {
            'Data Aktual (Transaksi Riil)': 32.5,
            'Data Imputasi (Forward Fill)': 67.5
        }
        
        fig_missing = create_pie_chart(
            labels=list(missing_data.keys()),
            values=list(missing_data.values()),
            title="Komposisi Data: Aktual vs Imputasi",
            height=350,
            colors=[COLORS['primary'], COLORS['secondary']]
        )
        st.plotly_chart(fig_missing, use_container_width=True, config={'displayModeBar': False})
    
    # Important note about forward fill
    st.warning("""
        **⚠️ Catatan Penting - Forward Fill Methodology:**
        
        Missing values pada hari non-trading (weekend/libur) di-handle dengan **Forward Fill per komoditas**:
        - Harga hari libur = Harga hari kerja terakhir
        - **Realistis** untuk konteks pasar tradisional (pedagang tidak mengubah harga saat pasar tutup)
        - Forward fill dilakukan **setelah train-test split** untuk mencegah data leakage
        - Evaluasi model menggunakan **dual-layer**: Global (termasuk ffill) vs Riil (hanya hari transaksi)
    """)
    
    # Evaluation metrics comparison
    st.markdown("#### Impact pada Evaluasi Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
            **✅ Layer 2: Hari Transaksi Riil (BERMAKNA)**
            - N = 429 samples (2.8% test set)
            - MAPE = **9.34%** (Sangat Baik)
            - R² = **0.941**
            - MAE = Rp 3,599.33
            
            → Ini adalah metrik **genuine** yang menunjukkan performa sesungguhnya
        """)
    
    with col2:
        st.info("""
            **📊 Layer 1: Global termasuk FFill (OVER-OPTIMISTIC)**
            - N = 15,276 samples (100% test set)
            - MAPE = **0.56%** (Terlalu bagus)
            - R² = **0.9998**
            - MAE = Rp 227.64
            
            → Metrik ini **misleading** karena dominasi baris ffill (97.2%)
        """)
    
    st.markdown("---")
    
    # Per-commodity missing pattern (if single commodity selected)
    if len(selected_commodities) == 1:
        commodity = selected_commodities[0]
        comm_data = df[df['Komoditas'] == commodity].copy()
        
        st.markdown(f"#### Missing Pattern: {commodity}")
        
        # Calculate actual transaction days
        if 'Harga_Kemarin' in comm_data.columns:
            # Detect actual transaction: Harga != Harga_Kemarin
            comm_data['Is_Transaction'] = (comm_data['Harga'] != comm_data['Harga_Kemarin'])
            
            n_total = len(comm_data)
            n_transactions = comm_data['Is_Transaction'].sum()
            n_ffill = n_total - n_transactions
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Hari", f"{n_total:,}")
            with col2:
                st.metric("Transaksi Riil", f"{n_transactions:,}", 
                         delta=f"{(n_transactions/n_total*100):.1f}%")
            with col3:
                st.metric("Forward Fill", f"{n_ffill:,}",
                         delta=f"{(n_ffill/n_total*100):.1f}%")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #64748B; font-size: 0.85rem;'>
        <p>Analisis EDA • PANTAU PASAR Dashboard</p>
    </div>
""", unsafe_allow_html=True)
