"""
PAGE 5: METODOLOGI & DOKUMENTASI AKADEMIS
- Pipeline Data & Hierarki Resolusi
- Autoregressive Feature Engineering
- Model Hyperparameters
- TimeSeriesSplit Validation
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.visualization import COLORS

# Page config
st.set_page_config(page_title="Metodologi - PANTAU PASAR", page_icon="📚", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .method-section {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .code-box {
        background-color: #1E293B;
        color: #E2E8F0;
        padding: 1rem;
        border-radius: 6px;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        overflow-x: auto;
    }
    .formula {
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
        <h1 style='color: #1E40AF; margin-bottom: 0.3rem;'>📚 Metodologi & Dokumentasi</h1>
        <p style='font-size: 1rem; color: #475569;'>
            Pipeline Data, Feature Engineering, dan Arsitektur Model
        </p>
    </div>
""", unsafe_allow_html=True)

# Table of Contents
st.markdown("## 📋 Daftar Isi")
st.markdown("""
1. [Pipeline Data & Hierarki Resolusi](#section-1)
2. [Autoregressive Feature Engineering](#section-2)
3. [Model Architecture & Hyperparameters](#section-3)
4. [TimeSeriesSplit Validation](#section-4)
5. [Evaluasi Dual-Layer](#section-5)
6. [Limitasi & Rekomendasi](#section-6)
""")

st.markdown("---")

# === SECTION 1: PIPELINE DATA ===
st.markdown("<div id='section-1'></div>", unsafe_allow_html=True)
st.markdown("## 1️⃣ Pipeline Data & Hierarki Resolusi")

st.markdown("""
<div class='method-section'>
    <h3 style='color: #1E40AF; margin-top: 0;'>1.1 Sumber Data Heterogen</h3>
    <p style='color: #475569; line-height: 1.8;'>
        Sistem mengintegrasikan data dari <strong>4 sumber berbeda</strong> dengan format dan frekuensi update yang berbeda:
    </p>
</div>
""", unsafe_allow_html=True)

# Data sources table
sources_data = {
    'Sumber': ['SP2KP', 'PIHPS', 'Diskoperindag', 'UPT Seketeng'],
    'Rows (Raw)': ['24,445', '20,940', '7,963', '601'],
    'Format': ['Wide (2 baris metadata)', 'Wide (tanggal spasi)', '2 layout berbeda', 'Long (tanggal kurung siku)'],
    'Prioritas': [1, 4, 2, 3],
    'Kontribusi Final': ['61.7%', '30.5%', '7.0%', '0.7%']
}
sources_df = pd.DataFrame(sources_data)

st.dataframe(sources_df, use_container_width=True, hide_index=True)

st.markdown("""
<div class='method-section'>
    <h3 style='color: #1E40AF; margin-top: 0;'>1.2 Hierarki Resolusi Konflik</h3>
    <p style='color: #475569; line-height: 1.8;'>
        Ketika kombinasi <code>(Tanggal, Komoditas)</code> sama muncul di multiple sumber, 
        sistem menggunakan <strong>hierarchical priority resolution</strong>:
    </p>
</div>
""", unsafe_allow_html=True)

st.code("""
# Algoritma Resolusi Konflik
1. Assign priority weight per sumber:
   - SP2KP (Kementerian Perdagangan) = Priority 1 (Highest)
   - Diskoperindag (Dinas Kabupaten) = Priority 2
   - UPT Seketeng (Unit Pasar) = Priority 3
   - PIHPS (Portal Umum) = Priority 4 (Lowest)

2. Sort data: [Tanggal↑, Komoditas↑, Priority↑]

3. Drop duplicates: keep='first' (prioritas tertinggi)

# Hasil:
- Input: 53,949 baris (4 sumber)
- Output: 39,601 baris (setelah deduplikasi)
- Konflik resolved: 14,348 duplikat (-26.6%)
""", language='python')

st.markdown("""
<div class='method-section'>
    <h3 style='color: #1E40AF; margin-top: 0;'>1.3 Calendar Grid Expansion</h3>
    <p style='color: #475569; line-height: 1.8;'>
        Model time series membutuhkan data <strong>kontinu setiap hari</strong> untuk lag/rolling features. 
        Data mentah hanya berisi hari trading (weekend/libur = missing).
    </p>
</div>
""", unsafe_allow_html=True)

st.code("""
# Master Kalender Grid
Master Kalender (2021-01-01 ... 2026-11-10)
   ↓  × Komoditas (57 unik)
   ↓  = 2,140 hari × 57 komoditas
   ↓  = 121,980 baris grid
   ↓  Left Join dengan data gabungan (39,601)
   →  preprocessed_dataset.csv

# Pentingnya Kontinuitas:
❌ Tanpa grid: Lag-7 bisa melompati 10+ hari kalender (weekend/libur)
✅ Dengan grid: Lag-7 = tepat 7 hari sebelumnya
""", language='python')

st.info("""
**Hasil Pipeline ETL:**
- Raw (4 sumber): 53,949 baris
- Merged: 39,601 baris (deduplikasi -26.6%)
- Preprocessed (Grid): 121,980 baris (expansion)
- Feature Engineered: 76,363 baris (setelah dropna)
""")

st.markdown("---")

# === SECTION 2: FEATURE ENGINEERING ===
st.markdown("<div id='section-2'></div>", unsafe_allow_html=True)
st.markdown("## 2️⃣ Autoregressive Feature Engineering")

st.markdown("""
<div class='method-section'>
    <h3 style='color: #1E40AF; margin-top: 0;'>2.1 Forward-Fill Imputation</h3>
    <p style='color: #475569; line-height: 1.8;'>
        Missing values pada hari non-trading di-handle dengan <strong>Forward Fill per komoditas</strong>:
    </p>
</div>
""", unsafe_allow_html=True)

st.code("""
# Forward Fill per Komoditas
df_sorted = df.sort_values(['Komoditas', 'Tanggal'])
df['Harga'] = df.groupby('Komoditas')['Harga'].ffill()

# Rationale:
# - Pasar tutup di weekend/libur → harga tidak berubah
# - Pedagang jual di harga hari terakhir
# - Realistis untuk konteks retail tradisional
""", language='python')

st.markdown("""
<div class='method-section'>
    <h3 style='color: #1E40AF; margin-top: 0;'>2.2 Fitur Temporal (13 Fitur)</h3>
</div>
""", unsafe_allow_html=True)

# Feature categories
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **📅 Kalender (6 fitur)**
    - `Tahun`: 2021-2026
    - `Bulan`: 1-12
    - `Hari`: 1-31
    - `DayOfWeek`: 0-6
    - `Quarter`: 1-4
    - `WeekOfYear`: 1-53
    """)

with col2:
    st.markdown("""
    **⏮️ Lag (2 fitur)**
    - `Harga_Kemarin`: shift(1)
    - `Harga_Minggu_Lalu`: shift(7)
    
    **Formula:**
    ```
    Lag_n[t] = Harga[t-n]
    ```
    """)

with col3:
    st.markdown("""
    **📊 Rolling (5 fitur)**
    - `Rolling_Mean_7`: rolling(7).mean()
    - `Rolling_Mean_14`: rolling(14).mean()
    - `Rolling_Std_7`: rolling(7).std()
    - `Rolling_Max_7`: rolling(7).max()
    - `Rolling_Min_7`: rolling(7).min()
    """)

st.markdown("""
<div class='formula'>
    <strong>Formula Rolling Features:</strong><br><br>
    <code>Rolling_Mean_7[t] = (Σ Harga[t-7:t-1]) / 7</code><br>
    <code>Rolling_Std_7[t] = std(Harga[t-7:t-1])</code><br>
    <code>Rolling_Max_7[t] = max(Harga[t-7:t-1])</code><br>
    <code>Rolling_Min_7[t] = min(Harga[t-7:t-1])</code><br><br>
    <strong>⚠️ Penting:</strong> Window di-shift 1 hari untuk mencegah target leakage
</div>
""", unsafe_allow_html=True)

st.warning("""
**🔍 Mengapa TIDAK Ada Variabel External?**

Variabel external seperti **cuaca, harga BBM, inflasi nasional** tidak dimasukkan karena:
1. **Data gap**: Data historis cuaca/BBM tidak tersedia lengkap untuk 5.8 tahun
2. **Granularity mismatch**: Inflasi nasional (monthly) vs prediksi harian
3. **Fokus baseline**: Penelitian ini fokus pada **autoregressive pure** untuk baseline
4. **Rekomendasi**: Penelitian lanjutan bisa augmentasi dengan external variables
""")

st.markdown("---")

# === SECTION 3: MODEL ARCHITECTURE ===
st.markdown("<div id='section-3'></div>", unsafe_allow_html=True)
st.markdown("## 3️⃣ Model Architecture & Hyperparameters")

st.markdown("""
<div class='method-section'>
    <h3 style='color: #1E40AF; margin-top: 0;'>3.1 Random Forest Regressor</h3>
    <p style='color: #475569; line-height: 1.8;'>
        <strong>Algorithm:</strong> RandomForestRegressor (scikit-learn 1.5+)<br><br>
        <strong>Alasan Pilihan:</strong>
    </p>
    <ul style='color: #475569; line-height: 1.8;'>
        <li>✅ Handle non-linear relationships</li>
        <li>✅ Robust terhadap outliers (cabai, tomat spikes)</li>
        <li>✅ Built-in feature importance (interpretability)</li>
        <li>✅ No assumption tentang distribusi data</li>
        <li>✅ Parallel training (n_jobs=-1)</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Hyperparameters table
st.markdown("### 3.2 Hyperparameter Tuning Results")

st.info("""
**Metode Tuning:** 2-Tahap GridSearchCV + TimeSeriesSplit (5 folds)
- **Tahap 1:** Coarse grid search (n_estimators, max_depth)
- **Tahap 2:** Fine-tuning (min_samples_split, min_samples_leaf)
- **Best CV MAE:** Rp 1,107.96
""")

hyperparam_data = {
    'Parameter': [
        'n_estimators',
        'max_depth',
        'min_samples_split',
        'min_samples_leaf',
        'random_state',
        'n_jobs'
    ],
    'Nilai': [100, 10, 12, 5, 42, -1],
    'Keterangan': [
        'Jumlah decision trees dalam ensemble',
        'Kedalaman maksimum setiap tree',
        'Minimum samples untuk split internal node',
        'Minimum samples di leaf node',
        'Seed untuk reproducibility',
        'Parallel processing (gunakan semua CPU cores)'
    ]
}
hyperparam_df = pd.DataFrame(hyperparam_data)

st.dataframe(hyperparam_df, use_container_width=True, hide_index=True)

st.markdown("### 3.3 Data Split Strategy")

st.code("""
# Chronological 80:20 Split (NO random shuffle)
# WHY: Time series → test harus setelah train (mencegah future leakage)

target_n_train = int(n_total * 0.80)
cum_per_date = df.groupby('Tanggal').size().cumsum()
cutoff_date = cum_per_date[cum_per_date <= target_n_train].idxmax()

train = df[df['Tanggal'] <= cutoff_date]
test = df[df['Tanggal'] > cutoff_date]

# Guard Check: PASS
assert test['Tanggal'].min() > train['Tanggal'].max()
""", language='python')

split_data = {
    'Set': ['Train', 'Test'],
    'Baris': ['61,087', '15,276'],
    'Persentase': ['80.0%', '20.0%'],
    'Rentang Tanggal': ['2021-01-18 → 2026-02-15', '2026-02-16 → 2026-11-10']
}
split_df = pd.DataFrame(split_data)

st.dataframe(split_df, use_container_width=True, hide_index=True)

st.markdown("---")

# === SECTION 4: TIME SERIES VALIDATION ===
st.markdown("<div id='section-4'></div>", unsafe_allow_html=True)
st.markdown("## 4️⃣ TimeSeriesSplit Validation")

st.markdown("""
<div class='method-section'>
    <h3 style='color: #1E40AF; margin-top: 0;'>4.1 Cross-Validation untuk Time Series</h3>
    <p style='color: #475569; line-height: 1.8;'>
        Berbeda dengan K-Fold standar, <strong>TimeSeriesSplit</strong> memastikan:
    </p>
    <ul style='color: #475569; line-height: 1.8;'>
        <li>✅ Train set selalu sebelum validation set (no future leakage)</li>
        <li>✅ Expanding window: fold ke-n memiliki lebih banyak data training</li>
        <li>✅ Simulasi prediksi real-time (model belajar dari masa lalu)</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.code("""
from sklearn.model_selection import TimeSeriesSplit

# 5-Fold TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    
    model.fit(X_train_fold, y_train_fold)
    score = evaluate(model, X_val_fold, y_val_fold)
    
# Hasil: Best CV MAE = Rp 1,107.96
""", language='python')

st.markdown("---")

# === SECTION 5: DUAL-LAYER EVALUATION ===
st.markdown("<div id='section-5'></div>", unsafe_allow_html=True)
st.markdown("## 5️⃣ Evaluasi Dual-Layer (Kontribusi Metodologis)")

st.markdown("""
<div class='method-section'>
    <h3 style='color: #1E40AF; margin-top: 0;'>5.1 Problem Discovery</h3>
    <p style='color: #475569; line-height: 1.8;'>
        Test set terdiri dari <strong>2 jenis baris</strong> dengan karakteristik berbeda:
    </p>
</div>
""", unsafe_allow_html=True)

layer_data = {
    'Layer': ['Hari FFill (Imputasi)', 'Hari Transaksi Riil'],
    'Jumlah Baris': ['14,847', '429'],
    'Persentase': ['97.2%', '2.8%'],
    'Karakteristik': [
        'Harga = Harga_Kemarin (pasar tutup)',
        'Harga ≠ Harga_Kemarin (pasar aktif)'
    ],
    'MAPE': ['0.56% (over-optimistic)', '9.34% (genuine)']
}
layer_df = pd.DataFrame(layer_data)

st.dataframe(layer_df, use_container_width=True, hide_index=True)

st.warning("""
**⚠️ Implikasi Metodologis:**

Jika evaluasi hanya menggunakan **Layer 1 (Global)**, metrik akan **misleading** karena:
1. Model hanya perlu "meng-copy" `Harga_Kemarin` untuk 97.2% test set
2. MAPE 0.56% terlihat "sempurna" tapi **bukan** indikasi performa sesungguhnya
3. Tidak merepresentasikan kemampuan model pada hari trading aktif

**Solusi:** Dual-Layer Evaluation
- **Layer 1 (Global):** Untuk transparansi, tapi dengan disclaimer
- **Layer 2 (Hari Riil):** Evaluasi genuine pada 429 hari transaksi aktif
""")

st.markdown("""
<div class='method-section'>
    <h3 style='color: #1E40AF; margin-top: 0;'>5.2 Metrik Evaluasi</h3>
</div>
""", unsafe_allow_html=True)

metrics_formula = """
**MAE (Mean Absolute Error):**
```
MAE = (1/n) Σ |y_actual - y_predicted|
```
→ Rata-rata kesalahan absolut dalam Rupiah

**RMSE (Root Mean Squared Error):**
```
RMSE = √[(1/n) Σ (y_actual - y_predicted)²]
```
→ Memberikan penalti lebih besar untuk error ekstrem

**MAPE (Mean Absolute Percentage Error):**
```
MAPE = (1/n) Σ |(y_actual - y_predicted) / y_actual| × 100%
```
→ Error dalam persentase (scale-invariant)

**R² (Coefficient of Determination):**
```
R² = 1 - (SS_res / SS_tot)
```
→ Proporsi variance yang dijelaskan model (0-1)
"""

st.markdown(metrics_formula)

# MAPE Category
st.markdown("### 5.3 Kategori MAPE Akademis")

mape_category = {
    'Kategori': ['Sangat Baik', 'Baik', 'Cukup', 'Buruk'],
    'Range MAPE': ['< 10%', '10% - 20%', '20% - 50%', '≥ 50%'],
    'Interpretasi': [
        'Model sangat akurat untuk decision-making',
        'Model cukup akurat untuk perencanaan',
        'Model perlu improvement signifikan',
        'Model tidak reliable untuk praktis'
    ],
    'Jumlah Komoditas': [12, 11, 0, 0]
}
mape_cat_df = pd.DataFrame(mape_category)

st.dataframe(mape_cat_df, use_container_width=True, hide_index=True)

st.success("""
**✅ Hasil Penelitian:**
- **23/23 komoditas** mencapai MAPE < 20% (kategori "Baik" atau lebih baik)
- **12/23 komoditas** mencapai MAPE < 10% (kategori "Sangat Baik")
- **0 komoditas** dalam kategori "Cukup" atau "Buruk"
- Model RF mengalahkan naive baseline di **17/23 komoditas** (73.9%)
""")

st.markdown("---")

# === SECTION 6: LIMITATIONS & RECOMMENDATIONS ===
st.markdown("<div id='section-6'></div>", unsafe_allow_html=True)
st.markdown("## 6️⃣ Limitasi & Rekomendasi Penelitian Lanjutan")

# Limitations
st.markdown("### 🔍 Limitasi Penelitian")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **📊 Data Limitations:**
    - ⚠️ Missing values: 67.5% raw calendar days
    - ⚠️ Coverage: Hanya Pasar Seketeng (tidak seluruh Sumbawa)
    - ⚠️ Period: 2021-2026 (belum capture long-term cycles 10+ tahun)
    - ⚠️ Imbalance: 97.2% test set adalah ffill
    
    **🤖 Model Limitations:**
    - ⚠️ No external variables (cuaca, BBM, inflasi)
    - ⚠️ Weak seasonality capture (fitur kalender terlalu simpel)
    - ⚠️ Cold start problem (komoditas baru tidak bisa diprediksi)
    - ⚠️ Outlier sensitivity (spike ekstrem >3σ sulit diprediksi)
    """)

with col2:
    st.markdown("""
    **📏 Evaluation Limitations:**
    - ⚠️ Threshold arbitrary: Min 5 baris riil per komoditas
    - ⚠️ Short horizon: Prediksi 1 hari ke depan only
    - ⚠️ No confidence interval (point estimate tanpa uncertainty)
    - ⚠️ 34 komoditas tidak memenuhi threshold evaluasi
    
    **🔬 Methodology Limitations:**
    - ⚠️ Autoregressive pure (no external features)
    - ⚠️ Single model for all commodities (bukan per-commodity)
    - ⚠️ No ensemble (RF only, no stacking/voting)
    """)

st.markdown("---")

# Recommendations
st.markdown("### 🚀 Rekomendasi Penelitian Lanjutan")

tabs = st.tabs(["Model Enhancement", "Feature Augmentation", "Data Collection", "Deployment"])

with tabs[0]:
    st.markdown("""
    **🔹 Ensemble Methods:**
    - Stacking: RF + XGBoost + LightGBM + Linear Regression (meta-learner)
    - Voting Regressor dengan weighted average berdasarkan per-commodity MAPE
    - Blend predictions dari multiple models untuk reduce variance
    
    **🔹 Deep Learning:**
    - **LSTM/GRU** untuk menangkap long-term dependencies (>14 hari)
    - **Temporal Fusion Transformer (TFT)** dengan attention mechanism
    - **Prophet** (Facebook) untuk seasonality decomposition
    
    **🔹 Hierarchical Modeling:**
    - Model global → model per-cluster komoditas (volatile vs stable)
    - Separate models untuk komoditas dengan karakteristik berbeda
    - Per-market models jika ekspansi ke pasar lain
    """)

with tabs[1]:
    st.markdown("""
    **🔹 External Variables:**
    - **Cuaca:** Suhu, curah hujan → pengaruh supply sayuran (cabai, tomat)
    - **Harga BBM:** Cost produksi dan transportasi
    - **Inflasi Nasional:** Purchasing power masyarakat
    - **Sentimen Berita:** Scraping berita ekonomi → sentiment analysis
    
    **🔹 Calendar Events:**
    - Binary flags: `is_ramadan`, `is_tahun_baru`, `is_idul_fitri`
    - Days-until-event: `days_to_ramadan` (countdown feature)
    - Day-after-event: `days_after_idul_fitri` (demand normalization)
    
    **🔹 Spatial Features:**
    - Harga komoditas di kabupaten tetangga (Bima, Lombok)
    - Distance to production center (km dari sentra produksi)
    - Supply volume data (jika tersedia dari dinas pertanian)
    """)

with tabs[2]:
    st.markdown("""
    **🔹 Frekuensi Lebih Tinggi:**
    - Real-time intraday (pagi vs sore) → capture daily volatility
    - Differentiate weekday vs weekend (beberapa kios buka weekend)
    - Hourly data untuk komoditas sangat volatile (cabai, tomat)
    
    **🔹 Granularity Lebih Detail:**
    - Varietas spesifik: Beras IR-64 vs Ciherang vs Slyp
    - Grade kualitas: A, B, C untuk protein (ayam, telur)
    - Ukuran/berat standar: Cabai rawit kualitas super vs biasa
    
    **🔹 Coverage Geografis:**
    - Pasar-pasar kecamatan (bukan hanya Seketeng)
    - Retail modern (supermarket, minimarket) → price comparison
    - Sentra produksi langsung (petani, peternak)
    """)

with tabs[3]:
    st.markdown("""
    **🔹 Dashboard Interaktif (Streamlit):**
    - ✅ **Sudah ada:** Dashboard PANTAU PASAR ini
    - Real-time prediction update via API integration
    - Alert system untuk anomaly detection (harga >2σ)
    - Historical comparison visualization per komoditas
    
    **🔹 Mobile App:**
    - Push notification untuk trader (harga prediksi besok pagi)
    - Price tracking per komoditas favorit (watchlist)
    - QR code scanner untuk check harga real-time vs prediksi
    
    **🔹 API Service:**
    - RESTful API untuk integrasi third-party (POS, e-commerce)
    - Batch prediction endpoint untuk policy simulation
    - WebSocket untuk real-time streaming predictions
    
    **🔹 Monitoring & Retraining:**
    - Model performance drift detection (MAPE naik >15%)
    - Automated retraining pipeline (monthly/quarterly)
    - A/B testing untuk model baru vs production model
    - MLOps integration (MLflow, DVC untuk version control)
    """)

st.markdown("---")

# Footer
st.markdown("### 📖 Referensi & Citation")

st.code("""
Saputra Budiman. (2026). Prediksi Harga Pangan Kabupaten Sumbawa 
Menggunakan Random Forest Regressor dengan Dual-Layer Evaluation. 
[Skripsi]. Sistem Informasi Manajemen, [Universitas].

Tech Stack:
- Python 3.14.3
- scikit-learn 1.5+
- pandas 2.2+, numpy 2.0+
- Streamlit 1.30+ (Dashboard)
- Plotly (Visualizations)
""", language='text')

st.info("""
**📌 Kontribusi Ilmiah:**
1. **Pipeline ETL robust** untuk data heterogen 4 sumber
2. **Hierarchical priority resolution** (SP2KP > Diskoperindag > UPT > PIHPS)
3. **Calendar grid expansion** untuk kontinuitas time series
4. **Dual-layer evaluation** (metodologi baru): Segmentasi riil vs imputasi
5. **Per-commodity analysis** dengan 23 komoditas evaluasi komprehensif
""")

# Final footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #64748B; font-size: 0.85rem; padding: 1rem 0;'>
        <p style='margin: 0;'>
            <strong>PANTAU PASAR</strong> - Sistem Pendukung Keputusan Harga Pangan<br>
            Pasar Seketeng, Kabupaten Sumbawa • 2026
        </p>
    </div>
""", unsafe_allow_html=True)
