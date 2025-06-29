# Step 1: Agricultural Association Rules Project Setup
# File: setup_project.py

import os
import subprocess
import sys
import pandas as pd
import numpy as np

def create_project_structure():
    """
    Create the complete project directory structure
    """
    directories = [
        'agricultural_association_rules',
        'agricultural_association_rules/data',
        'agricultural_association_rules/data/raw',
        'agricultural_association_rules/data/processed',
        'agricultural_association_rules/data/external',
        'agricultural_association_rules/notebooks',
        'agricultural_association_rules/src',
        'agricultural_association_rules/src/data',
        'agricultural_association_rules/src/features',
        'agricultural_association_rules/src/models',
        'agricultural_association_rules/src/visualization',
        'agricultural_association_rules/tests',
        'agricultural_association_rules/results',
        'agricultural_association_rules/results/figures',
        'agricultural_association_rules/results/models',
        'agricultural_association_rules/results/reports',
        'agricultural_association_rules/docs',
        'agricultural_association_rules/configs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    return directories

def create_requirements_file():
    """
    Create requirements.txt with all necessary dependencies
    """
    requirements = """
# Core Data Science Libraries
pandas>=1.5.0
numpy>=1.21.0
scipy>=1.9.0
scikit-learn>=1.1.0

# Association Rules Mining
mlxtend>=0.21.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.10.0
bokeh>=2.4.0

# Geospatial Analysis
geopandas>=0.11.0
folium>=0.12.0
rasterio>=1.3.0
earthpy>=0.9.0

# Machine Learning Enhancement
xgboost>=1.6.0
lightgbm>=3.3.0
catboost>=1.1.0

# Statistical Analysis
statsmodels>=0.13.0
pingouin>=0.5.0

# Data Processing
openpyxl>=3.0.0
xlrd>=2.0.0

# API and Data Collection
requests>=2.28.0
beautifulsoup4>=4.11.0
pyowm>=3.3.0  # OpenWeatherMap API

# Jupyter and Development
jupyter>=1.0.0
jupyterlab>=3.4.0
ipywidgets>=7.7.0

# Testing
pytest>=7.1.0
pytest-cov>=3.0.0

# Documentation
sphinx>=5.0.0
sphinx-rtd-theme>=1.0.0

# Configuration Management
pyyaml>=6.0
python-dotenv>=0.20.0

# Progress Bars and Utilities
tqdm>=4.64.0
rich>=12.5.0
"""
    
    with open('agricultural_association_rules/requirements.txt', 'w') as f:
        f.write(requirements.strip())
    
    print("✓ Created requirements.txt")
    return requirements

def create_environment_file():
    """
    Create .env file for environment variables
    """
    env_content = """
# Agricultural Association Rules Project Configuration
PROJECT_NAME=agricultural_association_rules
PROJECT_VERSION=1.0.0

# Data Directories
RAW_DATA_DIR=data/raw
PROCESSED_DATA_DIR=data/processed
EXTERNAL_DATA_DIR=data/external

# Model Configuration
MIN_SUPPORT=0.05
MIN_CONFIDENCE=0.6
MIN_LIFT=1.2

# API Keys (add your own)
OPENWEATHER_API_KEY=your_openweather_api_key_here
USDA_API_KEY=your_usda_api_key_here

# Database Configuration (if using)
DATABASE_URL=sqlite:///agricultural_data.db

# Logging Level
LOG_LEVEL=INFO
"""
    
    with open('agricultural_association_rules/.env', 'w') as f:
        f.write(env_content.strip())
    
    print("✓ Created .env file")
    return env_content

def create_config_file():
    """
    Create main configuration file
    """
    config_content = """
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
"""
    
    with open('agricultural_association_rules/configs/config.py', 'w') as f:
        f.write(config_content.strip())
    
    print("✓ Created config.py")
    return config_content

def create_init_files():
    """
    Create __init__.py files for Python packages
    """
    init_files = [
        'agricultural_association_rules/__init__.py',
        'agricultural_association_rules/src/__init__.py',
        'agricultural_association_rules/src/data/__init__.py',
        'agricultural_association_rules/src/features/__init__.py',
        'agricultural_association_rules/src/models/__init__.py',
        'agricultural_association_rules/src/visualization/__init__.py',
        'agricultural_association_rules/tests/__init__.py',
        'agricultural_association_rules/configs/__init__.py'
    ]
    
    for init_file in init_files:
        with open(init_file, 'w') as f:
            f.write('# -*- coding: utf-8 -*-\n')
        print(f"✓ Created {init_file}")

def create_readme():
    """
    Create project README.md
    """
    readme_content = """
# Agricultural Association Rules Project

## Overview
This project implements Association Rules mining for agricultural crop pattern analysis and precision farming applications.

## Project Structure
```
agricultural_association_rules/
├── data/
│   ├── raw/           # Original, immutable data
│   ├── processed/     # Cleaned and transformed data
│   └── external/      # External data sources
├── src/
│   ├── data/          # Data loading and preprocessing
│   ├── features/      # Feature engineering
│   ├── models/        # Association rules mining
│   └── visualization/ # Plotting and visualization
├── notebooks/         # Jupyter notebooks for exploration
├── tests/            # Unit tests
├── results/          # Generated analysis results
├── docs/             # Project documentation
└── configs/          # Configuration files
```

## Setup Instructions
1. Install Python 3.8+
2. Install dependencies: `pip install -r requirements.txt`
3. Configure environment variables in `.env`
4. Run initial data setup: `python setup_project.py`

## Usage
See notebooks/ directory for step-by-step analysis examples.

## Dependencies
See requirements.txt for complete list of dependencies.
"""
    
    with open('agricultural_association_rules/README.md', 'w') as f:
        f.write(readme_content.strip())
    
    print("✓ Created README.md")

def install_dependencies():
    """
    Install project dependencies
    """
    try:
        print("Installing dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "-r", "agricultural_association_rules/requirements.txt"
        ])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing dependencies: {e}")
        return False

def verify_installation():
    """
    Verify that key libraries are installed correctly
    """
    required_imports = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('mlxtend.frequent_patterns', 'apriori'),
        ('mlxtend.frequent_patterns', 'association_rules'),
        ('sklearn', None),
        ('matplotlib.pyplot', 'plt'),
        ('seaborn', 'sns')
    ]
    
    print("\nVerifying installation...")
    all_good = True
    
    for module, alias in required_imports:
        try:
            if alias:
                exec(f"import {module} as {alias}")
            else:
                exec(f"import {module}")
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            all_good = False
    
    return all_good

def main():
    """
    Main setup function
    """
    print("🌾 Setting up Agricultural Association Rules Project...")
    print("=" * 60)
    
    # Create project structure
    directories = create_project_structure()
    print()
    
    # Create configuration files
    create_requirements_file()
    create_environment_file()
    create_config_file()
    create_init_files()
    create_readme()
    print()
    
    # Install dependencies (optional - user can do manually)
    print("📦 Would you like to install dependencies now? (y/n)")
    install_deps = input().lower().strip() == 'y'
    
    if install_deps:
        success = install_dependencies()
        if success:
            verify_installation()
    else:
        print("⚠️  Remember to install dependencies with:")
        print("   pip install -r agricultural_association_rules/requirements.txt")
    
    print("\n" + "=" * 60)
    print("🎉 Project setup complete!")
    print("\nNext steps:")
    print("1. Navigate to the project directory: cd agricultural_association_rules")
    print("2. Install dependencies (if not done): pip install -r requirements.txt")
    print("3. Configure your .env file with API keys")
    print("4. Proceed to Step 2: Data Generation and Collection")
    print("=" * 60)

if __name__ == "__main__":
    main()