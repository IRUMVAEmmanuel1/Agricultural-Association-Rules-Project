# config.py - Main Configuration File

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"

# Association Rules Parameters
ASSOCIATION_RULES_CONFIG = {
    'min_support': float(os.getenv('MIN_SUPPORT', 0.05)),
    'min_confidence': float(os.getenv('MIN_CONFIDENCE', 0.6)),
    'min_lift': float(os.getenv('MIN_LIFT', 1.2)),
    'max_len': 6,  # Maximum itemset length
    'metric': 'confidence',
    'use_colnames': True
}

# Data Processing Parameters
DATA_CONFIG = {
    'soil_ph_bins': [0, 5.5, 7.0, 14],
    'soil_ph_labels': ['Acidic', 'Neutral', 'Alkaline'],
    'nitrogen_bins': [0, 20, 40, 100],
    'nitrogen_labels': ['Low', 'Medium', 'High'],
    'rainfall_bins': [0, 300, 800, 2000],
    'rainfall_labels': ['Low', 'Medium', 'High'],
    'temperature_bins': [0, 15, 25, 40],
    'temperature_labels': ['Cool', 'Moderate', 'Warm']
}

# Visualization Settings
PLOT_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'color_palette': 'viridis',
    'font_size': 12
}

# API Configuration
API_CONFIG = {
    'openweather_api_key': os.getenv('OPENWEATHER_API_KEY'),
    'usda_api_key': os.getenv('USDA_API_KEY'),
    'request_timeout': 30,
    'max_retries': 3
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': os.getenv('LOG_LEVEL', 'INFO'),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'agricultural_association_rules.log'
}