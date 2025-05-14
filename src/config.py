# File: src/config.py

import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
HISTORICAL_DATA_PATH = os.path.join(DATA_DIR, 'historical_data.csv')
FEAR_GREED_INDEX_PATH = os.path.join(DATA_DIR, 'fear_greed_index.csv')

# Output directory for saving results
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Analysis parameters
SENTIMENT_CATEGORIES = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']