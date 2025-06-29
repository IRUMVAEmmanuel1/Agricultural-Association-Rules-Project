# Step 2: Agricultural Dataset Generation
# File: src/data/generate_agricultural_data.py

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class AgriculturalDataGenerator:
    """
    Generate realistic agricultural datasets for association rules mining
    """
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
        # Define agricultural parameters
        self.crop_types = [
            'Corn', 'Soybeans', 'Wheat', 'Rice', 'Cotton', 'Tomatoes', 
            'Potatoes', 'Lettuce', 'Carrots', 'Onions', 'Beans', 'Peas'
        ]
        
        self.soil_types = ['Clay', 'Sandy', 'Loam', 'Silt', 'Clayey_Loam', 'Sandy_Loam']
        
        self.fertilizer_types = [
            'Nitrogen_High', 'Phosphorus_High', 'Potassium_High', 
            'NPK_Balanced', 'Organic_Compost', 'Urea', 'No_Fertilizer'
        ]
        
        self.irrigation_methods = [
            'Drip_Irrigation', 'Sprinkler', 'Flood_Irrigation', 
            'Rainfall_Only', 'Micro_Sprinkler', 'Subsurface_Drip'
        ]
        
        self.tillage_types = [
            'No_Till', 'Conventional_Till', 'Reduced_Till', 
            'Strip_Till', 'Deep_Till', 'Minimum_Till'
        ]
        
        self.regions = [
            'Midwest', 'Southeast', 'Northeast', 'Southwest', 
            'Pacific_Northwest', 'Great_Plains', 'California_Central_Valley'
        ]
        
        # Define realistic correlations
        self.crop_correlations = {
            'Corn': {'soil_preference': ['Loam', 'Silt'], 'fertilizer_preference': ['Nitrogen_High', 'NPK_Balanced']},
            'Rice': {'soil_preference': ['Clay', 'Clayey_Loam'], 'fertilizer_preference': ['NPK_Balanced']},
            'Tomatoes': {'soil_preference': ['Loam', 'Sandy_Loam'], 'fertilizer_preference': ['Phosphorus_High', 'NPK_Balanced']},
            'Potatoes': {'soil_preference': ['Sandy', 'Sandy_Loam'], 'fertilizer_preference': ['Potassium_High']},
            'Soybeans': {'soil_preference': ['Loam', 'Silt'], 'fertilizer_preference': ['Phosphorus_High', 'Organic_Compost']}
        }
    
    def generate_farm_records(self, n_records=5000):
        """
        Generate comprehensive farm records dataset
        """
        print(f"🌾 Generating {n_records} farm records...")
        
        records = []
        for i in range(n_records):
            # Basic farm information
            farm_id = f"FARM_{i+1:05d}"
            field_id = f"FIELD_{farm_id}_{random.randint(1, 5)}"
            
            # Geographic information
            region = np.random.choice(self.regions)
            latitude = self._generate_latitude_by_region(region)
            longitude = self._generate_longitude_by_region(region)
            elevation = np.random.uniform(0, 1500)  # meters
            
            # Soil characteristics
            soil_type = np.random.choice(self.soil_types)
            soil_ph = self._generate_soil_ph(soil_type)
            organic_matter = self._generate_organic_matter(soil_type)
            nitrogen_content = self._generate_nitrogen_content(organic_matter)
            phosphorus_content = np.random.uniform(10, 80)
            potassium_content = np.random.uniform(50, 300)
            
            # Climate data
            avg_temperature = self._generate_temperature_by_region(region)
            annual_rainfall = self._generate_rainfall_by_region(region)
            humidity = np.random.uniform(40, 85)
            solar_radiation = np.random.uniform(15, 30)  # MJ/m²/day
            
            # Crop selection (influenced by soil and climate)
            crop_type = self._select_crop_by_conditions(soil_type, avg_temperature, annual_rainfall)
            crop_variety = self._generate_crop_variety(crop_type)
            
            # Farming practices
            fertilizer_type = self._select_fertilizer(crop_type, soil_ph, nitrogen_content)
            irrigation_method = self._select_irrigation(annual_rainfall, region)
            tillage_type = np.random.choice(self.tillage_types)
            
            # Planting and harvest dates
            planting_date, harvest_date = self._generate_farming_dates(crop_type, region)
            
            # Input quantities
            fertilizer_amount = self._generate_fertilizer_amount(fertilizer_type, crop_type)
            seed_rate = self._generate_seed_rate(crop_type)
            irrigation_frequency = self._generate_irrigation_frequency(irrigation_method, annual_rainfall)
            
            # Pest and disease management
            pest_pressure = np.random.choice(['Low', 'Medium', 'High'], p=[0.4, 0.4, 0.2])
            disease_pressure = np.random.choice(['Low', 'Medium', 'High'], p=[0.5, 0.3, 0.2])
            pesticide_used = self._determine_pesticide_use(pest_pressure, disease_pressure)
            
            # Calculate yield based on multiple factors
            yield_per_hectare = self._calculate_yield(
                crop_type, soil_ph, organic_matter, nitrogen_content,
                avg_temperature, annual_rainfall, fertilizer_type,
                irrigation_method, pest_pressure, disease_pressure
            )
            
            # Economic data
            cost_per_hectare = self._calculate_costs(
                fertilizer_type, fertilizer_amount, irrigation_method,
                tillage_type, pesticide_used, seed_rate
            )
            
            market_price = self._generate_market_price(crop_type, harvest_date)
            revenue_per_hectare = yield_per_hectare * market_price
            profit_per_hectare = revenue_per_hectare - cost_per_hectare
            
            # Sustainability metrics
            water_usage = self._calculate_water_usage(irrigation_method, irrigation_frequency, crop_type)
            carbon_footprint = self._calculate_carbon_footprint(tillage_type, fertilizer_type, irrigation_method)
            
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
                'planting_date': planting_date,
                'harvest_date': harvest_date,
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
            
            if (i + 1) % 1000 == 0:
                print(f"   Generated {i+1}/{n_records} records...")
        
        df = pd.DataFrame(records)
        print(f"✅ Generated {len(df)} farm records successfully!")
        return df
    
    def _generate_latitude_by_region(self, region):
        """Generate realistic latitude based on region"""
        region_coords = {
            'Midwest': (39, 49),
            'Southeast': (25, 37),
            'Northeast': (40, 47),
            'Southwest': (25, 37),
            'Pacific_Northwest': (42, 49),
            'Great_Plains': (35, 49),
            'California_Central_Valley': (35, 40)
        }
        min_lat, max_lat = region_coords[region]
        return np.random.uniform(min_lat, max_lat)
    
    def _generate_longitude_by_region(self, region):
        """Generate realistic longitude based on region"""
        region_coords = {
            'Midwest': (-104, -80),
            'Southeast': (-106, -75),
            'Northeast': (-80, -66),
            'Southwest': (-125, -94),
            'Pacific_Northwest': (-125, -110),
            'Great_Plains': (-104, -90),
            'California_Central_Valley': (-122, -118)
        }
        min_lon, max_lon = region_coords[region]
        return np.random.uniform(min_lon, max_lon)
    
    def _generate_soil_ph(self, soil_type):
        """Generate realistic soil pH based on soil type"""
        ph_ranges = {
            'Clay': (6.0, 7.5),
            'Sandy': (5.5, 7.0),
            'Loam': (6.0, 7.5),
            'Silt': (6.2, 7.8),
            'Clayey_Loam': (6.0, 7.3),
            'Sandy_Loam': (5.8, 7.2)
        }
        min_ph, max_ph = ph_ranges[soil_type]
        return np.random.uniform(min_ph, max_ph)
    
    def _generate_organic_matter(self, soil_type):
        """Generate organic matter percentage based on soil type"""
        om_ranges = {
            'Clay': (2.5, 6.0),
            'Sandy': (0.5, 2.5),
            'Loam': (2.0, 5.0),
            'Silt': (2.0, 4.5),
            'Clayey_Loam': (2.2, 5.5),
            'Sandy_Loam': (1.5, 3.5)
        }
        min_om, max_om = om_ranges[soil_type]
        return np.random.uniform(min_om, max_om)
    
    def _generate_nitrogen_content(self, organic_matter):
        """Generate nitrogen content based on organic matter"""
        # Higher organic matter typically means higher nitrogen
        base_nitrogen = organic_matter * 8 + np.random.uniform(-5, 15)
        return max(5, min(100, base_nitrogen))
    
    def _generate_temperature_by_region(self, region):
        """Generate average temperature by region"""
        temp_ranges = {
            'Midwest': (8, 12),
            'Southeast': (15, 21),
            'Northeast': (6, 12),
            'Southwest': (18, 24),
            'Pacific_Northwest': (9, 13),
            'Great_Plains': (10, 14),
            'California_Central_Valley': (16, 19)
        }
        min_temp, max_temp = temp_ranges[region]
        return np.random.uniform(min_temp, max_temp)
    
    def _generate_rainfall_by_region(self, region):
        """Generate annual rainfall by region"""
        rainfall_ranges = {
            'Midwest': (600, 1200),
            'Southeast': (800, 1600),
            'Northeast': (700, 1300),
            'Southwest': (200, 600),
            'Pacific_Northwest': (400, 2500),
            'Great_Plains': (300, 800),
            'California_Central_Valley': (150, 500)
        }
        min_rain, max_rain = rainfall_ranges[region]
        return np.random.uniform(min_rain, max_rain)
    
    def _select_crop_by_conditions(self, soil_type, temperature, rainfall):
        """Select crop based on environmental conditions"""
        # Define crop suitability
        if temperature < 10 and rainfall > 800:
            suitable_crops = ['Wheat', 'Peas', 'Carrots']
        elif temperature > 20 and rainfall < 500:
            suitable_crops = ['Cotton', 'Tomatoes', 'Onions']
        elif soil_type in ['Clay', 'Clayey_Loam'] and rainfall > 1000:
            suitable_crops = ['Rice', 'Soybeans']
        elif soil_type in ['Sandy', 'Sandy_Loam']:
            suitable_crops = ['Potatoes', 'Carrots', 'Lettuce']
        else:
            suitable_crops = ['Corn', 'Soybeans', 'Wheat', 'Beans']
        
        return np.random.choice(suitable_crops)
    
    def _generate_crop_variety(self, crop_type):
        """Generate crop variety"""
        varieties = {
            'Corn': ['Dent_Corn', 'Sweet_Corn', 'Flint_Corn'],
            'Soybeans': ['Maturity_Group_2', 'Maturity_Group_3', 'Maturity_Group_4'],
            'Wheat': ['Winter_Wheat', 'Spring_Wheat', 'Durum_Wheat'],
            'Rice': ['Long_Grain', 'Medium_Grain', 'Short_Grain'],
            'Tomatoes': ['Determinate', 'Indeterminate', 'Cherry'],
            'Potatoes': ['Russet', 'Red', 'Fingerling']
        }
        if crop_type in varieties:
            return np.random.choice(varieties[crop_type])
        return f"{crop_type}_Standard"
    
    def _select_fertilizer(self, crop_type, soil_ph, nitrogen_content):
        """Select fertilizer based on crop needs and soil conditions"""
        if crop_type in self.crop_correlations:
            preferences = self.crop_correlations[crop_type]['fertilizer_preference']
            
            # Adjust based on soil conditions
            if nitrogen_content < 20:
                if 'Nitrogen_High' not in preferences:
                    preferences = preferences + ['Nitrogen_High']
            
            if soil_ph < 6.0:
                # Acidic soil might need organic amendments
                if 'Organic_Compost' not in preferences:
                    preferences = preferences + ['Organic_Compost']
            
            return np.random.choice(preferences)
        else:
            return np.random.choice(self.fertilizer_types)
    
    def _select_irrigation(self, rainfall, region):
        """Select irrigation method based on rainfall and region"""
        if rainfall < 400:
            return np.random.choice(['Drip_Irrigation', 'Sprinkler', 'Micro_Sprinkler'])
        elif rainfall < 800:
            return np.random.choice(['Drip_Irrigation', 'Sprinkler', 'Subsurface_Drip', 'Rainfall_Only'])
        else:
            return np.random.choice(['Rainfall_Only', 'Subsurface_Drip'], p=[0.7, 0.3])
    
    def _generate_farming_dates(self, crop_type, region):
        """Generate planting and harvest dates"""
        # Simplified seasonal planting
        base_year = 2023
        
        planting_months = {
            'Corn': (3, 5),  # March to May
            'Soybeans': (4, 6),  # April to June
            'Wheat': (9, 11),  # Fall planting
            'Rice': (3, 5),
            'Cotton': (3, 5),
            'Tomatoes': (3, 5),
            'Potatoes': (3, 4),
            'Lettuce': (2, 4),
            'Carrots': (3, 5),
            'Onions': (2, 4)
        }
        
        if crop_type in planting_months:
            start_month, end_month = planting_months[crop_type]
        else:
            start_month, end_month = (3, 5)
        
        planting_month = np.random.randint(start_month, end_month + 1)
        planting_day = np.random.randint(1, 29)
        planting_date = datetime(base_year, planting_month, planting_day)
        
        # Harvest typically 90-150 days after planting
        growing_days = np.random.randint(90, 151)
        harvest_date = planting_date + timedelta(days=growing_days)
        
        return planting_date.strftime('%Y-%m-%d'), harvest_date.strftime('%Y-%m-%d')
    
    def _generate_fertilizer_amount(self, fertilizer_type, crop_type):
        """Generate fertilizer application amount"""
        base_amounts = {
            'Nitrogen_High': 150,
            'Phosphorus_High': 100,
            'Potassium_High': 120,
            'NPK_Balanced': 130,
            'Organic_Compost': 2000,
            'Urea': 200,
            'No_Fertilizer': 0
        }
        base = base_amounts.get(fertilizer_type, 100)
        return base * np.random.uniform(0.7, 1.3)
    
    def _generate_seed_rate(self, crop_type):
        """Generate seed rate per hectare"""
        seed_rates = {
            'Corn': 25,
            'Soybeans': 60,
            'Wheat': 120,
            'Rice': 40,
            'Cotton': 20,
            'Tomatoes': 0.5,
            'Potatoes': 2500,
            'Lettuce': 2,
            'Carrots': 3,
            'Onions': 4
        }
        base_rate = seed_rates.get(crop_type, 50)
        return base_rate * np.random.uniform(0.8, 1.2)
    
    def _generate_irrigation_frequency(self, irrigation_method, rainfall):
        """Generate irrigation frequency in days"""
        if irrigation_method == 'Rainfall_Only':
            return 0
        elif irrigation_method == 'Drip_Irrigation':
            return np.random.randint(1, 3)  # Daily to every 3 days
        elif irrigation_method == 'Sprinkler':
            return np.random.randint(3, 7)  # Every 3-7 days
        else:
            return np.random.randint(5, 14)  # Weekly to bi-weekly
    
    def _determine_pesticide_use(self, pest_pressure, disease_pressure):
        """Determine if pesticides were used"""
        if pest_pressure == 'High' or disease_pressure == 'High':
            return np.random.choice(['Yes', 'No'], p=[0.8, 0.2])
        elif pest_pressure == 'Medium' or disease_pressure == 'Medium':
            return np.random.choice(['Yes', 'No'], p=[0.5, 0.5])
        else:
            return np.random.choice(['Yes', 'No'], p=[0.2, 0.8])
    
    def _calculate_yield(self, crop_type, soil_ph, organic_matter, nitrogen_content,
                        temperature, rainfall, fertilizer_type, irrigation_method,
                        pest_pressure, disease_pressure):
        """Calculate yield based on multiple factors"""
        
        # Base yields (tonnes per hectare)
        base_yields = {
            'Corn': 10.0,
            'Soybeans': 3.0,
            'Wheat': 5.0,
            'Rice': 8.0,
            'Cotton': 2.5,
            'Tomatoes': 60.0,
            'Potatoes': 45.0,
            'Lettuce': 25.0,
            'Carrots': 40.0,
            'Onions': 35.0,
            'Beans': 2.5,
            'Peas': 2.0
        }
        
        base_yield = base_yields.get(crop_type, 5.0)
        yield_multiplier = 1.0
        
        # Soil pH impact
        if 6.0 <= soil_ph <= 7.5:
            yield_multiplier *= 1.0
        elif 5.5 <= soil_ph < 6.0 or 7.5 < soil_ph <= 8.0:
            yield_multiplier *= 0.9
        else:
            yield_multiplier *= 0.7
        
        # Organic matter impact
        if organic_matter > 3.0:
            yield_multiplier *= 1.1
        elif organic_matter < 1.5:
            yield_multiplier *= 0.9
        
        # Nitrogen impact
        if nitrogen_content > 40:
            yield_multiplier *= 1.1
        elif nitrogen_content < 20:
            yield_multiplier *= 0.8
        
        # Fertilizer impact
        if fertilizer_type != 'No_Fertilizer':
            yield_multiplier *= 1.1
        
        # Weather impact
        if 600 <= rainfall <= 1200 and 15 <= temperature <= 25:
            yield_multiplier *= 1.0
        elif rainfall < 400 or rainfall > 1800:
            yield_multiplier *= 0.7
        elif temperature < 5 or temperature > 35:
            yield_multiplier *= 0.6
        
        # Irrigation impact
        if irrigation_method in ['Drip_Irrigation', 'Micro_Sprinkler'] and rainfall < 600:
            yield_multiplier *= 1.2
        
        # Pest and disease impact
        pressure_impact = {
            'Low': 1.0,
            'Medium': 0.9,
            'High': 0.7
        }
        yield_multiplier *= pressure_impact[pest_pressure]
        yield_multiplier *= pressure_impact[disease_pressure]
        
        # Add some randomness
        yield_multiplier *= np.random.uniform(0.8, 1.2)
        
        return base_yield * yield_multiplier
    
    def _calculate_costs(self, fertilizer_type, fertilizer_amount, irrigation_method,
                        tillage_type, pesticide_used, seed_rate):
        """Calculate production costs per hectare"""
        
        cost = 0
        
        # Fertilizer costs
        fertilizer_costs = {
            'Nitrogen_High': 1.2,
            'Phosphorus_High': 1.5,
            'Potassium_High': 1.0,
            'NPK_Balanced': 1.3,
            'Organic_Compost': 0.3,
            'Urea': 1.1,
            'No_Fertilizer': 0
        }
        cost += fertilizer_amount * fertilizer_costs.get(fertilizer_type, 0.5)
        
        # Irrigation costs
        irrigation_costs = {
            'Drip_Irrigation': 200,
            'Sprinkler': 150,
            'Flood_Irrigation': 100,
            'Rainfall_Only': 0,
            'Micro_Sprinkler': 180,
            'Subsurface_Drip': 220
        }
        cost += irrigation_costs.get(irrigation_method, 0)
        
        # Tillage costs
        tillage_costs = {
            'No_Till': 50,
            'Conventional_Till': 150,
            'Reduced_Till': 100,
            'Strip_Till': 80,
            'Deep_Till': 200,
            'Minimum_Till': 75
        }
        cost += tillage_costs.get(tillage_type, 100)
        
        # Pesticide costs
        if pesticide_used == 'Yes':
            cost += np.random.uniform(80, 200)
        
        # Seed costs (simplified)
        cost += seed_rate * 2.5
        
        # Labor and equipment (base cost)
        cost += np.random.uniform(300, 600)
        
        return cost
    
    def _generate_market_price(self, crop_type, harvest_date):
        """Generate market price with seasonal variation"""
        # Base prices per tonne
        base_prices = {
            'Corn': 200,
            'Soybeans': 450,
            'Wheat': 250,
            'Rice': 400,
            'Cotton': 1500,
            'Tomatoes': 800,
            'Potatoes': 300,
            'Lettuce': 1200,
            'Carrots': 400,
            'Onions': 350
        }
        
        base_price = base_prices.get(crop_type, 300)
        
        # Add seasonal and random variation
        seasonal_factor = np.random.uniform(0.8, 1.3)
        
        return base_price * seasonal_factor
    
    def _calculate_water_usage(self, irrigation_method, irrigation_frequency, crop_type):
        """Calculate water usage in mm"""
        if irrigation_method == 'Rainfall_Only':
            return 0
        
        base_usage = {
            'Drip_Irrigation': 300,
            'Sprinkler': 450,
            'Flood_Irrigation': 800,
            'Micro_Sprinkler': 350,
            'Subsurface_Drip': 280
        }
        
        usage = base_usage.get(irrigation_method, 400)
        
        # Adjust for frequency
        if irrigation_frequency > 0:
            frequency_factor = max(0.5, 7.0 / irrigation_frequency)
            usage *= frequency_factor
        
        return usage * np.random.uniform(0.7, 1.3)
    
    def _calculate_carbon_footprint(self, tillage_type, fertilizer_type, irrigation_method):
        """Calculate carbon footprint in kg CO2 equivalent"""
        carbon = 0
        
        # Tillage impact
        tillage_carbon = {
            'No_Till': 100,
            'Conventional_Till': 250,
            'Reduced_Till': 175,
            'Strip_Till': 130,
            'Deep_Till': 300,
            'Minimum_Till': 150
        }
        carbon += tillage_carbon.get(tillage_type, 200)
        
        # Fertilizer impact
        fertilizer_carbon = {
            'Nitrogen_High': 300,
            'Phosphorus_High': 150,
            'Potassium_High': 100,
            'NPK_Balanced': 250,
            'Organic_Compost': 50,
            'Urea': 350,
            'No_Fertilizer': 0
        }
        carbon += fertilizer_carbon.get(fertilizer_type, 150)
        
        # Irrigation impact
        irrigation_carbon = {
            'Drip_Irrigation': 80,
            'Sprinkler': 120,
            'Flood_Irrigation': 40,
            'Rainfall_Only': 0,
            'Micro_Sprinkler': 100,
            'Subsurface_Drip': 90
        }
        carbon += irrigation_carbon.get(irrigation_method, 60)
        
        return carbon * np.random.uniform(0.8, 1.2)

def generate_sample_datasets():
    """
    Generate multiple sample datasets for association rules mining
    """
    print("🌾 Starting Agricultural Dataset Generation...")
    print("=" * 60)
    
    generator = AgriculturalDataGenerator(seed=42)
    
    # Generate main farm records dataset
    farm_data = generator.generate_farm_records(n_records=5000)
    
    # Save to CSV
    output_dir = Path("agricultural_association_rules/data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    farm_data.to_csv(output_dir / "farm_records.csv", index=False)
    print(f"📁 Saved farm_records.csv with {len(farm_data)} records")
    
    # Generate summary statistics
    print("\n📊 Dataset Summary:")
    print("-" * 40)
    print(f"Total farms: {farm_data['farm_id'].nunique()}")
    print(f"Total fields: {farm_data['field_id'].nunique()}")
    print(f"Crops grown: {', '.join(farm_data['crop_type'].unique())}")
    print(f"Regions covered: {', '.join(farm_data['region'].unique())}")
    print(f"Average yield: {farm_data['yield_per_hectare'].mean():.2f} tonnes/ha")
    print(f"Average profit: ${farm_data['profit_per_hectare'].mean():.2f}/ha")
    
    # Display sample data
    print("\n📋 Sample Records:")
    print("-" * 40)
    sample_cols = ['crop_type', 'soil_type', 'fertilizer_type', 'irrigation_method', 
                   'yield_per_hectare', 'profit_per_hectare']
    print(farm_data[sample_cols].head(10).to_string(index=False))
    
    # Save metadata
    metadata = {
        'dataset_info': {
            'name': 'Agricultural Association Rules Dataset',
            'generated_date': datetime.now().isoformat(),
            'total_records': len(farm_data),
            'total_farms': farm_data['farm_id'].nunique(),
            'total_fields': farm_data['field_id'].nunique()
        },
        'column_descriptions': {
            'farm_id': 'Unique farm identifier',
            'field_id': 'Unique field identifier within farm',
            'region': 'Geographic region',
            'latitude': 'Field latitude coordinate',
            'longitude': 'Field longitude coordinate',
            'elevation': 'Field elevation in meters',
            'soil_type': 'Primary soil classification',
            'soil_ph': 'Soil pH level (acidity/alkalinity)',
            'organic_matter_percent': 'Percentage of organic matter in soil',
            'nitrogen_content': 'Available nitrogen in soil (ppm)',
            'phosphorus_content': 'Available phosphorus in soil (ppm)',
            'potassium_content': 'Available potassium in soil (ppm)',
            'avg_temperature': 'Average growing season temperature (°C)',
            'annual_rainfall': 'Total annual rainfall (mm)',
            'humidity_percent': 'Average relative humidity (%)',
            'solar_radiation': 'Average solar radiation (MJ/m²/day)',
            'crop_type': 'Primary crop grown',
            'crop_variety': 'Specific variety of crop',
            'planting_date': 'Date when crop was planted',
            'harvest_date': 'Date when crop was harvested',
            'fertilizer_type': 'Type of fertilizer applied',
            'fertilizer_amount_kg_ha': 'Fertilizer application rate (kg/hectare)',
            'irrigation_method': 'Irrigation system used',
            'irrigation_frequency_days': 'Days between irrigation cycles',
            'tillage_type': 'Soil preparation method',
            'seed_rate_kg_ha': 'Seeding rate (kg/hectare)',
            'pest_pressure': 'Level of pest infestation (Low/Medium/High)',
            'disease_pressure': 'Level of disease pressure (Low/Medium/High)',
            'pesticide_used': 'Whether pesticides were applied (Yes/No)',
            'yield_per_hectare': 'Crop yield (tonnes/hectare)',
            'cost_per_hectare': 'Total production cost ($/hectare)',
            'revenue_per_hectare': 'Total revenue ($/hectare)',
            'profit_per_hectare': 'Net profit ($/hectare)',
            'water_usage_mm': 'Irrigation water applied (mm)',
            'carbon_footprint_kg_co2': 'Carbon footprint (kg CO2 equivalent)'
        },
        'value_ranges': {
            'soil_ph': [farm_data['soil_ph'].min(), farm_data['soil_ph'].max()],
            'yield_per_hectare': [farm_data['yield_per_hectare'].min(), farm_data['yield_per_hectare'].max()],
            'profit_per_hectare': [farm_data['profit_per_hectare'].min(), farm_data['profit_per_hectare'].max()],
            'water_usage_mm': [farm_data['water_usage_mm'].min(), farm_data['water_usage_mm'].max()],
            'carbon_footprint_kg_co2': [farm_data['carbon_footprint_kg_co2'].min(), farm_data['carbon_footprint_kg_co2'].max()]
        },
        'categorical_values': {
            'crop_types': list(farm_data['crop_type'].unique()),
            'soil_types': list(farm_data['soil_type'].unique()),
            'fertilizer_types': list(farm_data['fertilizer_type'].unique()),
            'irrigation_methods': list(farm_data['irrigation_method'].unique()),
            'tillage_types': list(farm_data['tillage_type'].unique()),
            'regions': list(farm_data['region'].unique())
        }
    }
    
    # Save metadata as JSON
    with open(output_dir / "dataset_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"📁 Saved dataset metadata")
    
    # Create regional subsets for specialized analysis
    print("\n🗺️ Creating Regional Datasets...")
    for region in farm_data['region'].unique():
        regional_data = farm_data[farm_data['region'] == region]
        if len(regional_data) >= 100:  # Only create subset if sufficient data
            filename = f"farm_records_{region.lower().replace(' ', '_')}.csv"
            regional_data.to_csv(output_dir / filename, index=False)
            print(f"   📁 Saved {filename} with {len(regional_data)} records")
    
    # Create crop-specific datasets
    print("\n🌱 Creating Crop-Specific Datasets...")
    major_crops = farm_data['crop_type'].value_counts().head(5).index
    for crop in major_crops:
        crop_data = farm_data[farm_data['crop_type'] == crop]
        filename = f"farm_records_{crop.lower().replace(' ', '_')}.csv"
        crop_data.to_csv(output_dir / filename, index=False)
        print(f"   📁 Saved {filename} with {len(crop_data)} records")
    
    print("\n" + "=" * 60)
    print("✅ Dataset Generation Complete!")
    print("\nGenerated Files:")
    print("📁 data/raw/farm_records.csv - Main dataset")
    print("📁 data/raw/dataset_metadata.json - Data documentation")
    print("📁 data/raw/farm_records_[region].csv - Regional subsets")
    print("📁 data/raw/farm_records_[crop].csv - Crop-specific subsets")
    print("\n🎯 Ready for Step 3: Data Preprocessing and Categorization")
    print("=" * 60)
    
    return farm_data, metadata

def validate_dataset(df):
    """
    Validate the generated dataset for quality and completeness
    """
    print("\n🔍 Validating Dataset Quality...")
    print("-" * 40)
    
    validation_results = {}
    
    # Check for missing values
    missing_values = df.isnull().sum()
    validation_results['missing_values'] = missing_values[missing_values > 0].to_dict()
    
    if len(validation_results['missing_values']) == 0:
        print("✅ No missing values found")
    else:
        print(f"⚠️  Missing values detected: {validation_results['missing_values']}")
    
    # Check for realistic value ranges
    validations = []
    
    # Soil pH should be between 3-12
    ph_valid = df['soil_ph'].between(3, 12).all()
    validations.append(("Soil pH range (3-12)", ph_valid))
    
    # Yields should be positive
    yield_valid = (df['yield_per_hectare'] > 0).all()
    validations.append(("Positive yields", yield_valid))
    
    # Costs should be positive
    cost_valid = (df['cost_per_hectare'] > 0).all()
    validations.append(("Positive costs", cost_valid))
    
    # Coordinates should be reasonable for agricultural areas
    lat_valid = df['latitude'].between(20, 60).all()
    lon_valid = df['longitude'].between(-130, -60).all()
    validations.append(("Latitude range (20-60°N)", lat_valid))
    validations.append(("Longitude range (130-60°W)", lon_valid))
    
    # Display validation results
    for validation_name, is_valid in validations:
        status = "✅" if is_valid else "❌"
        print(f"{status} {validation_name}")
        validation_results[validation_name] = is_valid
    
    # Statistical summary
    print("\n📊 Statistical Summary:")
    print("-" * 40)
    numeric_cols = ['yield_per_hectare', 'profit_per_hectare', 'soil_ph', 'organic_matter_percent']
    print(df[numeric_cols].describe().round(2))
    
    return validation_results

def create_quick_visualization(df):
    """
    Create quick visualizations to verify data quality
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        print("\n📈 Creating Quick Visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Agricultural Dataset Overview', fontsize=16, fontweight='bold')
        
        # 1. Crop type distribution
        crop_counts = df['crop_type'].value_counts()
        axes[0, 0].pie(crop_counts.values, labels=crop_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Crop Type Distribution')
        
        # 2. Yield distribution by crop
        df.boxplot(column='yield_per_hectare', by='crop_type', ax=axes[0, 1])
        axes[0, 1].set_title('Yield Distribution by Crop Type')
        axes[0, 1].set_xlabel('Crop Type')
        axes[0, 1].set_ylabel('Yield (tonnes/ha)')
        plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Soil pH vs Yield scatter plot
        scatter = axes[1, 0].scatter(df['soil_ph'], df['yield_per_hectare'], 
                                   c=df['organic_matter_percent'], cmap='viridis', alpha=0.6)
        axes[1, 0].set_xlabel('Soil pH')
        axes[1, 0].set_ylabel('Yield (tonnes/ha)')
        axes[1, 0].set_title('Soil pH vs Yield (colored by Organic Matter %)')
        plt.colorbar(scatter, ax=axes[1, 0])
        
        # 4. Regional distribution
        region_counts = df['region'].value_counts()
        axes[1, 1].bar(range(len(region_counts)), region_counts.values)
        axes[1, 1].set_xticks(range(len(region_counts)))
        axes[1, 1].set_xticklabels(region_counts.index, rotation=45)
        axes[1, 1].set_title('Records by Region')
        axes[1, 1].set_ylabel('Number of Records')
        
        plt.tight_layout()
        
        # Save the plot
        output_dir = Path("agricultural_association_rules/results/figures")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "dataset_overview.png", dpi=300, bbox_inches='tight')
        print(f"📁 Saved visualization: results/figures/dataset_overview.png")
        plt.close()
        
    except ImportError:
        print("⚠️  Matplotlib not available for visualization")
    except Exception as e:
        print(f"⚠️  Visualization error: {e}")

def main():
    """
    Main function to generate agricultural datasets
    """
    try:
        # Generate the datasets
        farm_data, metadata = generate_sample_datasets()
        
        # Validate the data
        validation_results = validate_dataset(farm_data)
        
        # Create visualizations
        create_quick_visualization(farm_data)
        
        # Save validation results
        output_dir = Path("agricultural_association_rules/data/raw")
        with open(output_dir / "validation_results.json", 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        print("\n🎯 Data Generation Summary:")
        print("=" * 50)
        print(f"✅ Main dataset: {len(farm_data)} records")
        print(f"✅ Validation: {'PASSED' if all(validation_results.values()) else 'ISSUES FOUND'}")
        print(f"✅ Files created in: {output_dir}")
        print("✅ Visualizations saved")
        
        # Display next steps
        print(f"\n📋 Next Steps:")
        print("1. Review the generated datasets in data/raw/")
        print("2. Check the visualization in results/figures/")
        print("3. Proceed to Step 3: Data Preprocessing")
        print("=" * 50)
        
        return farm_data, metadata, validation_results
        
    except Exception as e:
        print(f"❌ Error during dataset generation: {e}")
        import traceback
        traceback.print_exc()
        raise

def create_data_quality_report(farm_data, metadata, validation_results):
    """
    Create a comprehensive data quality report
    """
    print("\n📊 Creating Data Quality Report...")
    
    report = {
        'generation_timestamp': datetime.now().isoformat(),
        'dataset_summary': {
            'total_records': len(farm_data),
            'total_columns': len(farm_data.columns),
            'memory_usage_mb': round(farm_data.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        },
        'data_completeness': {
            'complete_records': len(farm_data.dropna()),
            'completeness_percentage': round((len(farm_data.dropna()) / len(farm_data)) * 100, 2)
        },
        'value_distributions': {
            'crop_types': farm_data['crop_type'].value_counts().to_dict(),
            'regions': farm_data['region'].value_counts().to_dict(),
            'soil_types': farm_data['soil_type'].value_counts().to_dict()
        },
        'statistical_summary': {
            'yield_stats': {
                'mean': round(farm_data['yield_per_hectare'].mean(), 2),
                'std': round(farm_data['yield_per_hectare'].std(), 2),
                'min': round(farm_data['yield_per_hectare'].min(), 2),
                'max': round(farm_data['yield_per_hectare'].max(), 2)
            },
            'profit_stats': {
                'mean': round(farm_data['profit_per_hectare'].mean(), 2),
                'std': round(farm_data['profit_per_hectare'].std(), 2),
                'min': round(farm_data['profit_per_hectare'].min(), 2),
                'max': round(farm_data['profit_per_hectare'].max(), 2)
            }
        },
        'validation_results': validation_results
    }
    
    # Save detailed report
    output_dir = Path("agricultural_association_rules/data/raw")
    with open(output_dir / "data_quality_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("✅ Data quality report saved")
    return report

def preview_association_rules_potential(farm_data):
    """
    Preview the potential for association rules mining
    """
    print("\n🔍 Previewing Association Rules Potential...")
    print("-" * 50)
    
    # Count unique values for key categorical variables
    categorical_vars = ['crop_type', 'soil_type', 'fertilizer_type', 'irrigation_method', 
                       'tillage_type', 'region', 'pest_pressure', 'disease_pressure']
    
    print("📊 Categorical Variable Diversity:")
    for var in categorical_vars:
        if var in farm_data.columns:
            unique_count = farm_data[var].nunique()
            print(f"   {var}: {unique_count} unique values")
    
    # Analyze potential transaction sizes
    print(f"\n📈 Transaction Potential:")
    print(f"   Average items per transaction: ~{len(categorical_vars)}")
    print(f"   Total possible combinations: Very high (good for mining)")
    
    # Show some example combinations
    print(f"\n🌾 Sample Farming Practice Combinations:")
    sample_combos = farm_data.groupby(['crop_type', 'soil_type', 'fertilizer_type']).size().head(5)
    for combo, count in sample_combos.items():
        print(f"   {combo[0]} + {combo[1]} + {combo[2]}: {count} occurrences")
    
    print("-" * 50)

if __name__ == "__main__":
    print("🌾 Agricultural Association Rules - Dataset Generation")
    print("=" * 60)
    print("This script generates realistic agricultural datasets for")
    print("association rules mining and precision farming analysis.")
    print("=" * 60)
    
# Add this to the end of your existing file after the last line:

if __name__ == "__main__":
    print("🌾 Agricultural Association Rules - Dataset Generation")
    print("=" * 60)
    print("This script generates realistic agricultural datasets for")
    print("association rules mining and precision farming analysis.")
    print("=" * 60)
    
    try:
        # Generate the agricultural datasets
        farm_data, metadata, validation_results = main()
        
        # Create detailed quality report
        quality_report = create_data_quality_report(farm_data, metadata, validation_results)
        
        # Preview association rules potential
        preview_association_rules_potential(farm_data)
        
        print(f"\n🎉 Step 2 Complete!")
        print("Ready to proceed to Step 3: Data Preprocessing and Categorization")
        
    except KeyboardInterrupt:
        print("\n⏹️  Generation interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print("Please check the error details above and try again.")
        exit(1)