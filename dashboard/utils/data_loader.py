"""
Data Loader Utility
Handles data loading with caching to prevent global data leakage
"""

import streamlit as st
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Base path untuk data (relative to dashboard folder)
BASE_PATH = Path(__file__).parent.parent.parent


@st.cache_data(ttl=3600)
def load_features_dataset() -> pd.DataFrame:
    """
    Load feature-engineered dataset (76,363 rows x 17 columns)
    
    Returns:
        DataFrame with columns: Tanggal, Komoditas, Harga, 13 features, Sumber
    """
    try:
        file_path = BASE_PATH / "processed_data" / "features" / "features_all_dataset.csv"
        df = pd.read_csv(file_path)
        df['Tanggal'] = pd.to_datetime(df['Tanggal'])
        return df
    except FileNotFoundError:
        st.error(f"⚠️ File tidak ditemukan: {file_path}")
        st.info("Generating sample data for demonstration...")
        return _generate_sample_data()


@st.cache_data(ttl=3600)
def load_merged_dataset() -> pd.DataFrame:
    """
    Load merged dataset from 4 sources (39,601 rows)
    
    Returns:
        DataFrame with columns: Tanggal, Komoditas, Harga, Sumber
    """
    try:
        file_path = BASE_PATH / "processed_data" / "merged" / "dataset_all_merged.csv"
        df = pd.read_csv(file_path)
        df['Tanggal'] = pd.to_datetime(df['Tanggal'])
        return df
    except FileNotFoundError:
        st.warning(f"⚠️ File tidak ditemukan: {file_path}")
        return _generate_sample_merged_data()


@st.cache_data(ttl=3600)
def load_model_artifacts() -> Dict:
    """
    Load trained Random Forest model and metadata
    
    Returns:
        Dictionary containing model, feature_columns, and metadata
    """
    try:
        model_path = BASE_PATH / "models" / "model_rf.pkl"
        feature_path = BASE_PATH / "models" / "feature_columns.json"
        metadata_path = BASE_PATH / "models" / "training_metadata.json"
        
        artifacts = {}
        
        # Load model
        with open(model_path, 'rb') as f:
            artifacts['model'] = pickle.load(f)
        
        # Load feature columns
        with open(feature_path, 'r') as f:
            artifacts['feature_columns'] = json.load(f)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            artifacts['metadata'] = json.load(f)
            
        return artifacts
    except FileNotFoundError as e:
        st.warning(f"⚠️ Model artifacts tidak lengkap: {e}")
        return _generate_mock_model_artifacts()


@st.cache_data(ttl=3600)
def load_evaluation_results() -> pd.DataFrame:
    """
    Load per-commodity evaluation results (23 commodities with MAPE < 20%)
    
    Returns:
        DataFrame with columns: Komoditas, MAPE, MAE, RMSE, R2, Kategori
    """
    try:
        file_path = BASE_PATH / "results" / "per_commodity_evaluation.csv"
        return pd.read_csv(file_path)
    except FileNotFoundError:
        return _generate_sample_evaluation()


@st.cache_data
def get_commodity_list(df: pd.DataFrame) -> List[str]:
    """Get unique commodity list sorted alphabetically"""
    return sorted(df['Komoditas'].unique().tolist())


@st.cache_data
def get_source_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate distribution of data sources from merged dataset
    
    Args:
        df: DataFrame with 'Sumber' column
        
    Returns:
        DataFrame with columns: Sumber, Jumlah, Persentase
    """
    if 'Sumber' not in df.columns:
        return pd.DataFrame(columns=['Sumber', 'Jumlah', 'Persentase'])
    
    source_counts = df['Sumber'].value_counts()
    total = len(df)
    
    result = pd.DataFrame({
        'Sumber': source_counts.index,
        'Jumlah': source_counts.values,
        'Persentase': (source_counts.values / total * 100)
    })
    
    return result.sort_values('Jumlah', ascending=False).reset_index(drop=True)


@st.cache_data
def calculate_commodity_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate volatility metrics for each commodity
    
    Args:
        df: DataFrame with 'Komoditas' and 'Harga' columns
        
    Returns:
        DataFrame with columns: Komoditas, Mean, Std, CV, Min, Max, Range
        Sorted by CV (Coefficient of Variation) descending
    """
    if 'Komoditas' not in df.columns or 'Harga' not in df.columns:
        return pd.DataFrame()
    
    volatility_data = []
    
    for commodity in df['Komoditas'].unique():
        comm_data = df[df['Komoditas'] == commodity]['Harga']
        
        if len(comm_data) > 1:  # Need at least 2 points for std
            mean_price = comm_data.mean()
            std_price = comm_data.std()
            cv = (std_price / mean_price * 100) if mean_price > 0 else 0
            
            volatility_data.append({
                'Komoditas': commodity,
                'Mean': mean_price,
                'Std': std_price,
                'CV': cv,
                'Min': comm_data.min(),
                'Max': comm_data.max(),
                'Range': comm_data.max() - comm_data.min()
            })
    
    result = pd.DataFrame(volatility_data)
    
    if not result.empty:
        result = result.sort_values('CV', ascending=False).reset_index(drop=True)
    
    return result


@st.cache_data
def get_data_source_distribution() -> Dict[str, float]:
    """
    Returns the data source distribution percentages (static/hardcoded)
    Based on analysis: SP2KP (61.7%), PIHPS (30.5%), Diskoperindag (7.0%), UPT Seketeng (0.7%)
    
    Note: This is a static function. Use get_source_distribution(df) for dynamic calculation.
    """
    return {
        'SP2KP': 61.7,
        'PIHPS': 30.5,
        'Diskoperindag': 7.0,
        'UPT Seketeng': 0.7
    }


@st.cache_data
def get_feature_importance() -> Dict[str, float]:
    """
    Returns feature importance based on analysis
    """
    return {
        'Harga_Kemarin': 83.29,
        'Rolling_Mean_7': 4.79,
        'Rolling_Mean_14': 4.08,
        'Rolling_Max_7': 3.49,
        'Rolling_Min_7': 2.98,
        'Harga_Minggu_Lalu': 1.36,
        'Tahun': 0.01,
        'Bulan': 0.00,
        'Hari': 0.00,
        'DayOfWeek': 0.00,
        'Quarter': 0.00,
        'WeekOfYear': 0.00,
        'Rolling_Std_7': 0.00
    }


@st.cache_data
def get_model_performance_metrics() -> Dict:
    """
    Returns dual-layer model performance metrics
    """
    return {
        'real_transactions': {
            'MAE': 3599.33,
            'RMSE': 5480.39,
            'MAPE': 9.34,
            'R2': 0.941,
            'n_samples': 429,
            'description': 'Metrik murni pada hari pasar aktif (transaksi riil)'
        },
        'global': {
            'MAE': 227.64,
            'RMSE': 1027.27,
            'MAPE': 0.56,
            'R2': 0.9998,
            'n_samples': 15276,
            'description': 'Metrik keseluruhan termasuk hari ffill (over-optimistic)'
        },
        'naive_baseline': {
            'MAE': 3655.19,
            'RMSE': 5549.89,
            'MAPE': 9.46,
            'description': 'Naive baseline: prediksi = Harga_Kemarin'
        }
    }


@st.cache_data
def get_top_volatile_commodities() -> pd.DataFrame:
    """
    Returns top 10 most volatile commodities
    CV = Coefficient of Variation (std/mean * 100%)
    """
    data = {
        'Komoditas': [
            'Cabai Rawit Merah', 'Tomat', 'Cabai Merah Besar', 
            'Cabai Merah Keriting', 'Bawang Merah', 'Bawang Putih Honan',
            'Daging Ayam Ras', 'Ikan Tongkol', 'Telur Ayam Ras', 'Gula Pasir Curah'
        ],
        'Mean (Rp)': [52139, 14835, 39941, 46272, 31537, 37889, 40833, 26111, 30667, 18824],
        'Std (Rp)': [24858, 6753, 13641, 15098, 8695, 9234, 3920, 4472, 1233, 984],
        'CV (%)': [47.67, 45.53, 34.15, 32.64, 27.58, 24.38, 9.60, 17.12, 4.02, 5.23],
        'Kategori': [
            'Sangat Volatile', 'Sangat Volatile', 'Volatile', 'Volatile',
            'Moderate Volatile', 'Moderate Volatile', 'Stabil', 'Moderate Volatile',
            'Sangat Stabil', 'Sangat Stabil'
        ]
    }
    return pd.DataFrame(data)


@st.cache_data
def get_commodity_statistics(df: pd.DataFrame, commodity: str) -> Dict:
    """
    Calculate statistics for a specific commodity
    """
    commodity_data = df[df['Komoditas'] == commodity]['Harga']
    
    if len(commodity_data) == 0:
        return None
    
    return {
        'mean': float(commodity_data.mean()),
        'median': float(commodity_data.median()),
        'std': float(commodity_data.std()),
        'min': float(commodity_data.min()),
        'max': float(commodity_data.max()),
        'cv': float((commodity_data.std() / commodity_data.mean()) * 100),
        'count': int(len(commodity_data))
    }


def filter_by_date_range(df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    """Filter DataFrame by date range"""
    mask = (df['Tanggal'] >= pd.to_datetime(start_date)) & (df['Tanggal'] <= pd.to_datetime(end_date))
    return df[mask]


def filter_by_commodity(df: pd.DataFrame, commodities: List[str]) -> pd.DataFrame:
    """Filter DataFrame by list of commodities"""
    return df[df['Komoditas'].isin(commodities)]


def filter_by_source(df: pd.DataFrame, sources: List[str]) -> pd.DataFrame:
    """Filter DataFrame by data sources"""
    if 'Sumber' not in df.columns:
        return df
    return df[df['Sumber'].isin(sources)]


# === SAMPLE DATA GENERATORS (for demo when real files not found) ===

def _generate_sample_data() -> pd.DataFrame:
    """Generate sample feature dataset for demonstration"""
    np.random.seed(42)
    dates = pd.date_range('2021-01-18', '2026-11-10', freq='D')
    commodities = ['Beras Medium', 'Cabai Rawit Merah', 'Minyak Goreng Sawit Curah', 
                   'Telur Ayam Ras', 'Gula Pasir Curah']
    
    data = []
    for commodity in commodities:
        base_price = {'Beras Medium': 14500, 'Cabai Rawit Merah': 52000, 
                      'Minyak Goreng Sawit Curah': 22500, 'Telur Ayam Ras': 30500,
                      'Gula Pasir Curah': 18800}[commodity]
        
        for date in dates:
            price = base_price + np.random.normal(0, base_price * 0.05)
            data.append({
                'Tanggal': date,
                'Komoditas': commodity,
                'Harga': price,
                'Harga_Kemarin': price * 0.99,
                'Harga_Minggu_Lalu': price * 0.98,
                'Rolling_Mean_7': price,
                'Rolling_Mean_14': price,
                'Rolling_Std_7': price * 0.02,
                'Rolling_Max_7': price * 1.05,
                'Rolling_Min_7': price * 0.95,
                'Tahun': date.year,
                'Bulan': date.month,
                'Hari': date.day,
                'DayOfWeek': date.dayofweek,
                'Quarter': date.quarter,
                'WeekOfYear': date.isocalendar()[1],
                'Sumber': np.random.choice(['SP2KP', 'PIHPS', 'Diskoperindag', 'UPT Seketeng'])
            })
    
    return pd.DataFrame(data)


def _generate_sample_merged_data() -> pd.DataFrame:
    """Generate sample merged dataset"""
    df = _generate_sample_data()
    return df[['Tanggal', 'Komoditas', 'Harga', 'Sumber']]


def _generate_mock_model_artifacts() -> Dict:
    """Generate mock model artifacts for demo"""
    return {
        'model': None,
        'feature_columns': [
            'Harga_Kemarin', 'Harga_Minggu_Lalu', 'Rolling_Mean_7', 'Rolling_Mean_14',
            'Rolling_Std_7', 'Rolling_Max_7', 'Rolling_Min_7', 'Tahun', 'Bulan',
            'Hari', 'DayOfWeek', 'Quarter', 'WeekOfYear'
        ],
        'metadata': {
            'model_type': 'RandomForestRegressor',
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 12,
            'min_samples_leaf': 5,
            'training_date': '2026-01-15'
        }
    }


def _generate_sample_evaluation() -> pd.DataFrame:
    """Generate sample per-commodity evaluation"""
    data = {
        'Komoditas': [
            'Daging Ayam Ras', 'Minyak Goreng Sawit Curah', 'Telur Ayam Ras',
            'Beras Medium', 'Beras Premium', 'Gula Pasir Curah', 'Udang Basah',
            'Minyak Goreng Premium', 'Kedelai Lokal', 'Bawang Putih Honan'
        ],
        'MAPE': [3.43, 3.49, 4.02, 4.35, 4.59, 5.23, 6.96, 7.63, 7.64, 7.96],
        'MAE': [1400, 785, 1233, 634, 716, 984, 4565, 1975, 1426, 3016],
        'RMSE': [1850, 1020, 1580, 820, 920, 1250, 5800, 2500, 1850, 3800],
        'R2': [0.95, 0.96, 0.94, 0.97, 0.96, 0.95, 0.91, 0.93, 0.94, 0.92],
        'Kategori': ['Sangat Baik'] * 10
    }
    return pd.DataFrame(data)
