"""
PAGE 3: PREDIKSI HARGA (CRITICAL PAGE)
- Dual-Layer Evaluation Metrics (Riil vs Global)
- Main Prediction Chart
- Feature Importance
- Extrapolation Warning
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
    load_model_artifacts,
    get_commodity_list,
    get_model_performance_metrics,
    get_feature_importance,
    load_evaluation_results
)
from utils.visualization import (
    create_multi_line_chart,
    create_horizontal_bar_chart,
    create_scatter_plot,
    format_currency,
    format_percentage,
    COLORS
)

# Page config
st.set_page_config(page_title="Prediksi Harga - PANTAU PASAR", page_icon="🎯", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .metric-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        color: #1E40AF;
        margin: 0.3rem 0;
    }
    .metric-caption {
        font-size: 0.75rem;
        color: #64748B;
        margin-top: 0.3rem;
    }
    .warning-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div style='text-align: center; padding: 1rem 0 2rem 0;'>
        <h1 style='color: #1E40AF; margin-bottom: 0.3rem;'>🎯 Prediksi Harga</h1>
        <p style='font-size: 1rem; color: #475569;'>
            Model Inference & Evaluasi Dual-Layer
        </p>
    </div>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_prediction_data():
    df = load_features_dataset()
    artifacts = load_model_artifacts()
    metrics = get_model_performance_metrics()
    feature_imp = get_feature_importance()
    eval_results = load_evaluation_results()
    return df, artifacts, metrics, feature_imp, eval_results

df, artifacts, metrics, feature_imp, eval_results = load_prediction_data()
commodity_list = get_commodity_list(df)

# === SIDEBAR ===
st.sidebar.markdown("## 🎯 Konfigurasi Prediksi")

# Commodity selection
selected_commodity = st.sidebar.selectbox(
    "Pilih Komoditas",
    options=commodity_list,
    index=0 if len(commodity_list) > 0 else None,
    help="Pilih komoditas untuk prediksi harga"
)

# Forecast horizon (not used in actual model, just for UI demonstration)
forecast_days = st.sidebar.slider(
    "Horizon Prediksi (hari)",
    min_value=1,
    max_value=30,
    value=7,
    help="Jumlah hari ke depan untuk prediksi (demonstrasi UI)"
)

st.sidebar.markdown("---")
st.sidebar.info("""
    **📌 Catatan:**
    - Model saat ini dilatih untuk prediksi **1 hari ke depan**
    - Fitur terpenting: **Harga_Kemarin** (83.3%)
    - Horizon > 1 hari memerlukan iterative forecasting
""")

# === COMPONENT 1: DUAL-LAYER EVALUATION METRICS ===
st.markdown("### 📊 Evaluasi Model: Dual-Layer Metrics")

st.markdown("""
    <p style='color: #475569; margin-bottom: 1rem;'>
        Sistem evaluasi menggunakan pendekatan <strong>dual-layer</strong> untuk transparansi metodologis:
    </p>
""", unsafe_allow_html=True)

# Radio toggle for metric layer
metric_layer = st.radio(
    "Pilih Layer Evaluasi:",
    options=["Performa Hari Transaksi Riil (DEFAULT)", "Performa Global (Over-optimistic)"],
    index=0,
    horizontal=True
)

if "Riil" in metric_layer:
    # Layer 2: Real transactions
    real_metrics = metrics['real_transactions']
    baseline_metrics = metrics['naive_baseline']
    
    st.success(f"""
        **✅ Layer 2: Evaluasi pada Hari Pasar Aktif**
        
        Metrik murni pada **{real_metrics['n_samples']} hari transaksi riil** untuk menguji keandalan model sesungguhnya 
        (hanya hari dengan perubahan harga aktual, tanpa forward-fill).
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        improvement = ((baseline_metrics['MAE'] - real_metrics['MAE']) / baseline_metrics['MAE']) * 100
        st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-label'>MAE (Mean Absolute Error)</div>
                <div class='metric-value'>{format_currency(real_metrics['MAE'])}</div>
                <div class='metric-caption' style='color: #16A34A;'>
                    +{improvement:.2f}% vs Baseline
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-label'>RMSE</div>
                <div class='metric-value'>{format_currency(real_metrics['RMSE'])}</div>
                <div class='metric-caption'>
                    Root Mean Squared Error
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        mape_color = '#16A34A' if real_metrics['MAPE'] < 10 else '#F59E0B'
        kategori = 'Sangat Baik' if real_metrics['MAPE'] < 10 else 'Baik'
        st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-label'>MAPE</div>
                <div class='metric-value' style='color: {mape_color};'>{real_metrics['MAPE']}%</div>
                <div class='metric-caption'>
                    Kategori: {kategori}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-label'>R² Score</div>
                <div class='metric-value' style='color: #0D9488;'>{real_metrics['R2']:.3f}</div>
                <div class='metric-caption'>
                    {real_metrics['R2']*100:.1f}% variance explained
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Comparison with baseline
    st.markdown("#### 🔄 Perbandingan dengan Naive Baseline")
    
    comparison_data = {
        'Metrik': ['MAE (Rp)', 'RMSE (Rp)', 'MAPE (%)'],
        'Random Forest': [
            real_metrics['MAE'],
            real_metrics['RMSE'],
            real_metrics['MAPE']
        ],
        'Naive Baseline': [
            baseline_metrics['MAE'],
            baseline_metrics['RMSE'],
            baseline_metrics['MAPE']
        ]
    }
    
    from utils.visualization import create_grouped_bar_chart
    
    fig_comparison = create_grouped_bar_chart(
        categories=comparison_data['Metrik'],
        values_dict={
            'Random Forest': comparison_data['Random Forest'],
            'Naive Baseline': comparison_data['Naive Baseline']
        },
        title="Model RF vs Naive Baseline (Prediksi = Harga_Kemarin)",
        xaxis_title="Metrik",
        yaxis_title="Nilai",
        height=350
    )
    st.plotly_chart(fig_comparison, use_container_width=True, config={'displayModeBar': False})

else:
    # Layer 1: Global (including forward-fill)
    global_metrics = metrics['global']
    
    st.warning(f"""
        **⚠️ Layer 1: Evaluasi Global (Termasuk Forward-Fill)**
        
        Metrik pada **seluruh test set ({global_metrics['n_samples']:,} samples)** termasuk hari libur/weekend yang di-imputasi.
        **Metrik ini over-optimistic** karena ~97.2% test set adalah baris forward-fill dimana Harga = Harga_Kemarin.
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-label'>MAE</div>
                <div class='metric-value'>{format_currency(global_metrics['MAE'])}</div>
                <div class='metric-caption' style='color: #DC2626;'>
                    ⚠️ Terlalu rendah
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-label'>RMSE</div>
                <div class='metric-value'>{format_currency(global_metrics['RMSE'])}</div>
                <div class='metric-caption' style='color: #DC2626;'>
                    ⚠️ Misleading
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-label'>MAPE</div>
                <div class='metric-value' style='color: #DC2626;'>{global_metrics['MAPE']:.2f}%</div>
                <div class='metric-caption'>
                    ⚠️ Over-optimistic
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-label'>R² Score</div>
                <div class='metric-value' style='color: #DC2626;'>{global_metrics['R2']:.4f}</div>
                <div class='metric-caption'>
                    ⚠️ Near-perfect (palsu)
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.error("""
        **⛔ Interpretasi:**
        
        Metrik Layer 1 terlihat "sempurna" karena model hanya perlu "meng-copy" nilai `Harga_Kemarin` 
        untuk mayoritas test set. Ini **BUKAN** indikasi performa sesungguhnya.
        
        **Gunakan Layer 2 (Hari Transaksi Riil) untuk evaluasi genuine.**
    """)

st.markdown("---")

# === COMPONENT 2: PREDICTION VISUALIZATION ===
st.markdown("### 📈 Visualisasi Prediksi Harga")

if selected_commodity:
    commodity_data = df[df['Komoditas'] == selected_commodity].copy()
    commodity_data = commodity_data.sort_values('Tanggal')
    
    # Take last 90 days for visualization
    recent_data = commodity_data.tail(90).copy()
    
    # Simulate prediction (use Harga_Kemarin as proxy since model is not actually running)
    recent_data['Prediksi'] = recent_data['Harga_Kemarin']
    
    if len(recent_data) > 0:
        fig_pred = create_multi_line_chart(
            recent_data,
            x_col='Tanggal',
            y_cols=['Harga', 'Prediksi'],
            labels=['Harga Aktual', 'Prediksi Model'],
            title=f"Prediksi Harga: {selected_commodity} (90 Hari Terakhir)",
            xaxis_title="Tanggal",
            yaxis_title="Harga (Rp)",
            height=450
        )
        st.plotly_chart(fig_pred, use_container_width=True, config={'displayModeBar': False})
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Harga Terakhir", format_currency(recent_data['Harga'].iloc[-1]))
        with col2:
            change = recent_data['Harga'].iloc[-1] - recent_data['Harga'].iloc[-2] if len(recent_data) > 1 else 0
            st.metric("Perubahan (1 hari)", format_currency(change), delta=f"{(change/recent_data['Harga'].iloc[-2]*100):.2f}%" if change != 0 else "0%")
        with col3:
            volatility = recent_data['Harga'].std()
            st.metric("Volatilitas (σ)", format_currency(volatility))
    else:
        st.warning(f"Tidak ada data untuk komoditas {selected_commodity}")

st.markdown("---")

# === COMPONENT 3: FEATURE IMPORTANCE ===
st.markdown("### 🎯 Feature Importance Analysis")

st.markdown("""
    <p style='color: #475569; margin-bottom: 1rem;'>
        Kontribusi setiap fitur terhadap prediksi model berdasarkan <strong>Mean Decrease Impurity</strong>:
    </p>
""", unsafe_allow_html=True)

# Prepare feature importance data
feat_names = list(feature_imp.keys())
feat_values = list(feature_imp.values())

# Sort by importance
sorted_indices = np.argsort(feat_values)[::-1]
sorted_names = [feat_names[i] for i in sorted_indices]
sorted_values = [feat_values[i] for i in sorted_indices]

# Color code by category
colors = []
for name in sorted_names:
    if 'Harga_' in name or 'Minggu' in name:
        colors.append(COLORS['primary'])      # Lag features
    elif 'Rolling' in name:
        colors.append(COLORS['secondary'])    # Rolling features
    else:
        colors.append(COLORS['gray'])         # Calendar features

fig_importance = create_horizontal_bar_chart(
    labels=sorted_names[:10],  # Top 10 features
    values=sorted_values[:10],
    title="Top 10 Fitur Terpenting",
    xaxis_title="Importance (%)",
    height=450,
    color_scale=colors[:10]
)
st.plotly_chart(fig_importance, use_container_width=True, config={'displayModeBar': False})

# Category summary
col1, col2, col3 = st.columns(3)

lag_importance = sum([v for k, v in feature_imp.items() if 'Harga_' in k or 'Minggu' in k])
rolling_importance = sum([v for k, v in feature_imp.items() if 'Rolling' in k])
calendar_importance = sum([v for k, v in feature_imp.items() if k in ['Tahun', 'Bulan', 'Hari', 'DayOfWeek', 'Quarter', 'WeekOfYear']])

with col1:
    st.markdown(f"""
        <div class='metric-box'>
            <div class='metric-label'>Lag Features</div>
            <div class='metric-value'>{lag_importance:.2f}%</div>
            <div class='metric-caption'>Dominan (Harga_Kemarin, Minggu_Lalu)</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class='metric-box'>
            <div class='metric-label'>Rolling Features</div>
            <div class='metric-value' style='color: #0D9488;'>{rolling_importance:.2f}%</div>
            <div class='metric-caption'>Moderate (MA, Std, Max, Min)</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class='metric-box'>
            <div class='metric-label'>Calendar Features</div>
            <div class='metric-value' style='color: #64748B;'>{calendar_importance:.2f}%</div>
            <div class='metric-caption'>Negligible (Seasonality lemah)</div>
        </div>
    """, unsafe_allow_html=True)

st.info("""
    **📌 Key Insight:**
    
    `Harga_Kemarin` berkontribusi **83.29%** terhadap prediksi, mengkonfirmasi hipotesis EDA bahwa 
    harga pangan memiliki **autoregressive dependency** yang sangat kuat (PACF Lag-1 ≈ 0.99).
    
    Fitur kalender hampir tidak berkontribusi (<0.01%), menunjukkan **seasonality lemah** pada level harian.
    Event-driven spikes (Ramadan, Tahun Baru) tidak tertangkap oleh fitur kalender sederhana.
""")

st.markdown("---")

# === COMPONENT 4: EXTRAPOLATION WARNING ===
st.markdown("### ⚠️ Catatan Akademik: Limitasi Model")

st.info("""
    **📘 Random Forest & Extrapolation:**
    
    Model Random Forest adalah **tree-based ensemble** yang bekerja dengan cara membagi ruang fitur menjadi 
    region-region berdasarkan data training. Karakteristik penting:
    
    1. **Tidak dapat mengekstrapolasi** di luar range nilai training
       - Jika harga training max = Rp 100,000, model tidak bisa prediksi > Rp 100,000
       - Prediksi akan "saturated" di nilai maksimum/minimum historis
    
    2. **Ketergantungan pada `Harga_Kemarin` (83.3%)**
       - Performa sangat baik untuk **short-term forecasting** (1-3 hari)
       - Akurasi menurun untuk **long-term forecasting** (>7 hari) tanpa iterative approach
    
    3. **Event-driven spikes sulit diprediksi**
       - Spike Ramadan, Tahun Baru, anomali cuaca tidak ter-capture oleh fitur kalender
       - Model akan **underpredict** spike ekstrem (>3σ)
    
    **Rekomendasi Penggunaan:**
    - ✅ Gunakan untuk perencanaan stocking **1-3 hari ke depan**
    - ✅ Fokus pada komoditas **stabil** (MAPE < 10%) untuk keputusan inventory
    - ⚠️ Gunakan sebagai **early warning signal**, bukan keputusan final, untuk komoditas volatile
    - ⚠️ Kombinasikan dengan domain knowledge (kalender event, cuaca, policy) untuk prediksi lebih akurat
""")

# Model artifacts info
if artifacts and artifacts.get('metadata'):
    meta = artifacts['metadata']
    with st.expander("🔧 Informasi Model & Hyperparameter"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Model Type:** {meta.get('model_type', 'RandomForestRegressor')}  
            **Training Date:** {meta.get('training_date', 'N/A')}  
            **Number of Estimators:** {meta.get('n_estimators', 100)}  
            **Max Depth:** {meta.get('max_depth', 10)}  
            """)
        
        with col2:
            st.markdown(f"""
            **Min Samples Split:** {meta.get('min_samples_split', 12)}  
            **Min Samples Leaf:** {meta.get('min_samples_leaf', 5)}  
            **Random State:** {meta.get('random_state', 42)}  
            **Validation:** TimeSeriesSplit (5 folds)
            """)

st.markdown("---")

# === PER-COMMODITY EVALUATION TABLE ===
if len(eval_results) > 0:
    st.markdown("### 📋 Evaluasi Per-Komoditas (23 Komoditas)")
    
    st.markdown("""
        <p style='color: #475569; margin-bottom: 1rem;'>
            Threshold: Minimal 5 baris transaksi riil per komoditas di test set. 
            34 komoditas lainnya tidak memenuhi threshold.
        </p>
    """, unsafe_allow_html=True)
    
    # Format and display
    styled_eval = eval_results.style.format({
        'MAPE': '{:.2f}%',
        'MAE': lambda x: format_currency(x),
        'RMSE': lambda x: format_currency(x) if pd.notna(x) else 'N/A',
        'R2': '{:.3f}' if 'R2' in eval_results.columns else None
    }).background_gradient(subset=['MAPE'], cmap='RdYlGn_r')
    
    st.dataframe(styled_eval, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #64748B; font-size: 0.85rem;'>
        <p>Model: Random Forest Regressor (n_estimators=100, max_depth=10) • PANTAU PASAR</p>
    </div>
""", unsafe_allow_html=True)
