# Quick Agricultural Data Generation Script
# File: quick_generate_data.py
# Save this in your agricultural_association_rules directory and run it

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def quick_generate_agricultural_data(n_records=1000):
    """
    Quick data generation for Step 2 missing data
    """
    print(f"🌾 Quick generating {n_records} agricultural records...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Define agricultural parameters
    crop_types = ['Corn', 'Soybeans', 'Wheat', 'Rice', 'Cotton', 'Tomatoes', 'Potatoes']
    soil_types = ['Clay', 'Sandy', 'Loam', 'Silt', 'Clayey_Loam', 'Sandy_Loam']
    fertilizer_types = ['Nitrogen_High', 'Phosphorus_High', 'NPK_Balanced', 'Organic_Compost', 'Urea']
    irrigation_methods = ['Drip_Irrigation', 'Sprinkler', 'Flood_Irrigation', 'Rainfall_Only']
    tillage_types = ['No_Till', 'Conventional_Till', 'Reduced_Till', 'Strip_Till']
    regions = ['Midwest', 'Southeast', 'Northeast', 'Southwest', 'Pacific_Northwest']
    
    records = []
    
    for i in range(n_records):
        # Basic information
        farm_id = f"FARM_{i+1:05d}"
        field_id = f"FIELD_{farm_id}_{random.randint(1, 3)}"
        
        # Location
        region = np.random.choice(regions)
        latitude = np.random.uniform(30, 50)
        longitude = np.random.uniform(-120, -70)
        elevation = np.random.uniform(0, 1000)
        
        # Soil properties
        soil_type = np.random.choice(soil_types)
        soil_ph = np.random.uniform(5.5, 8.0)
        organic_matter = np.random.uniform(1.0, 6.0)
        nitrogen_content = np.random.uniform(10, 80)
        phosphorus_content = np.random.uniform(15, 70)
        potassium_content = np.random.uniform(80, 250)
        
        # Climate
        avg_temperature = np.random.uniform(8, 25)
        annual_rainfall = np.random.uniform(300, 1500)
        humidity = np.random.uniform(45, 80)
        solar_radiation = np.random.uniform(16, 28)
        
        # Crop and practices
        crop_type = np.random.choice(crop_types)
        crop_variety = f"{crop_type}_Standard"
        fertilizer_type = np.random.choice(fertilizer_types)
        irrigation_method = np.random.choice(irrigation_methods)
        tillage_type = np.random.choice(tillage_types)
        
        # Dates
        planting_date = datetime(2023, random.randint(3, 5), random.randint(1, 28))
        harvest_date = planting_date + timedelta(days=random.randint(90, 150))
        
        # Quantities and management
        fertilizer_amount = np.random.uniform(50, 200)
        seed_rate = np.random.uniform(20, 100)
        irrigation_frequency = random.randint(0, 14)
        pest_pressure = np.random.choice(['Low', 'Medium', 'High'])
        disease_pressure = np.random.choice(['Low', 'Medium', 'High'])
        pesticide_used = np.random.choice(['Yes', 'No'])
        
        # Performance metrics
        base_yield = {'Corn': 9, 'Soybeans': 3, 'Wheat': 4, 'Rice': 7, 'Cotton': 2, 'Tomatoes': 50, 'Potatoes': 35}
        yield_multiplier = np.random.uniform(0.7, 1.4)
        yield_per_hectare = base_yield.get(crop_type, 5) * yield_multiplier
        
        cost_per_hectare = np.random.uniform(800, 2000)
        market_price = np.random.uniform(200, 1000)
        revenue_per_hectare = yield_per_hectare * market_price
        profit_per_hectare = revenue_per_hectare - cost_per_hectare
        
        # Environmental metrics
        water_usage = np.random.uniform(200, 600) if irrigation_method != 'Rainfall_Only' else 0
        carbon_footprint = np.random.uniform(200, 800)
        
        record = {
            'farm_id': farm_id,
            'field_id': field_id,
            'region': region,
            'latitude': round(latitude, 6),
            'longitude': round(longitude, 6),
            'elevation': round(elevation, 1),
            'soil_type': soil_type,
            'soil_ph': round(soil_ph, 2),
            'organic_matter_percent': round(organic_matter, 2),
            'nitrogen_content': round(nitrogen_content, 1),
            'phosphorus_content': round(phosphorus_content, 1),
            'potassium_content': round(potassium_content, 1),
            'avg_temperature': round(avg_temperature, 1),
            'annual_rainfall': round(annual_rainfall, 1),
            'humidity_percent': round(humidity, 1),
            'solar_radiation': round(solar_radiation, 1),
            'crop_type': crop_type,
            'crop_variety': crop_variety,
            'planting_date': planting_date.strftime('%Y-%m-%d'),
            'harvest_date': harvest_date.strftime('%Y-%m-%d'),
            'fertilizer_type': fertilizer_type,
            'fertilizer_amount_kg_ha': round(fertilizer_amount, 1),
            'irrigation_method': irrigation_method,
            'irrigation_frequency_days': irrigation_frequency,
            'tillage_type': tillage_type,
            'seed_rate_kg_ha': round(seed_rate, 1),
            'pest_pressure': pest_pressure,
            'disease_pressure': disease_pressure,
            'pesticide_used': pesticide_used,
            'yield_per_hectare': round(yield_per_hectare, 2),
            'cost_per_hectare': round(cost_per_hectare, 2),
            'revenue_per_hectare': round(revenue_per_hectare, 2),
            'profit_per_hectare': round(profit_per_hectare, 2),
            'water_usage_mm': round(water_usage, 1),
            'carbon_footprint_kg_co2': round(carbon_footprint, 1)
        }
        
        records.append(record)
        
        if (i + 1) % 200 == 0:
            print(f"   Generated {i+1}/{n_records} records...")
    
    return pd.DataFrame(records)

def main():
    """
    Quick data generation main function
    """
    print("🚀 Quick Agricultural Data Generation")
    print("=" * 50)
    
    # Generate data
    df = quick_generate_agricultural_data(1000)
    
    # Create directories
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save data
    output_file = output_dir / "farm_records.csv"
    df.to_csv(output_file, index=False)
    
    print(f"✅ Generated {len(df)} records")
    print(f"📁 Saved to: {output_file}")
    print(f"📊 Sample data:")
    print(df[['crop_type', 'soil_type', 'yield_per_hectare', 'profit_per_hectare']].head())
    print("\n🎯 Now you can run Step 3 preprocessing!")
    print("   python src/data/preprocess_agricultural_data.py")
    
    return df

if __name__ == "__main__":
    main()