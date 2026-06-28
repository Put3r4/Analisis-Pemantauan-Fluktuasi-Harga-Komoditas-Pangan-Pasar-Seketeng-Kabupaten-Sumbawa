import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from utils.data_loader import calculate_commodity_volatility, load_features_dataset

df = load_features_dataset()
vol = calculate_commodity_volatility(df)
print(f'✓ Volatility calculated: {len(vol)} commodities')
print(f'  Top 3 volatile: {", ".join(vol.head(3)["Komoditas"].tolist())}')
print(f'  CV range: {vol["CV"].min():.2f}% - {vol["CV"].max():.2f}%')
