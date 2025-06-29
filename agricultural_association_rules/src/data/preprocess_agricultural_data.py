
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AgriculturalDataPreprocessor:
    """
    Preprocess agricultural data for association rules mining
    """
    
    def __init__(self, config_file=None):
        # Load configuration
        if config_file:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self._default_config()
        
        # Define categorization rules
        self.categorization_rules = {
            'soil_ph': {
                'bins': [0, 5.5, 7.0, 14],
                'labels': ['Acidic', 'Neutral', 'Alkaline'],
                'prefix': 'pH'
            },
            'organic_matter_percent': {
                'bins': [0, 2, 4, 10],
                'labels': ['Low', 'Medium', 'High'],
                'prefix': 'OrganicMatter'
            },
            'nitrogen_content': {
                'bins': [0, 20, 40, 100],
                'labels': ['Low', 'Medium', 'High'],
                'prefix': 'Nitrogen'
            },
            'phosphorus_content': {
                'bins': [0, 25, 50, 100],
                'labels': ['Low', 'Medium', 'High'],
                'prefix': 'Phosphorus'
            },
            'potassium_content': {
                'bins': [0, 100, 200, 500],
                'labels': ['Low', 'Medium', 'High'],
                'prefix': 'Potassium'
            },
            'annual_rainfall': {
                'bins': [0, 400, 800, 2500],
                'labels': ['Low', 'Medium', 'High'],
                'prefix': 'Rainfall'
            },
            'avg_temperature': {
                'bins': [0, 15, 25, 40],
                'labels': ['Cool', 'Moderate', 'Warm'],
                'prefix': 'Temperature'
            },
            'yield_per_hectare': {
                'method': 'quartiles',
                'labels': ['Low', 'Medium', 'High', 'VeryHigh'],
                'prefix': 'Yield'
            },
            'profit_per_hectare': {
                'method': 'quartiles',
                'labels': ['Low', 'Medium', 'High', 'VeryHigh'],
                'prefix': 'Profit'
            },
            'water_usage_mm': {
                'bins': [0, 200, 400, 1000],
                'labels': ['Low', 'Medium', 'High'],
                'prefix': 'WaterUsage'
            },
            'carbon_footprint_kg_co2': {
                'bins': [0, 300, 600, 1000],
                'labels': ['Low', 'Medium', 'High'],
                'prefix': 'CarbonFootprint'
            }
        }
        
        # Define direct categorical columns (no transformation needed)
        self.direct_categorical = [
            'crop_type', 'soil_type', 'fertilizer_type', 'irrigation_method',
            'tillage_type', 'region', 'pest_pressure', 'disease_pressure',
            'pesticide_used', 'crop_variety'
        ]
    
    def _default_config(self):
        """Default configuration for preprocessing"""
        return {
            'min_support_threshold': 0.01,
            'enable_seasonal_features': True,
            'enable_economic_features': True,
            'enable_sustainability_features': True,
            'enable_geographic_features': True
        }
    
    def load_agricultural_data(self, file_path):
        """
        Load the agricultural dataset
        """
        print(f"📂 Loading agricultural data from {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Loaded {len(df)} records with {len(df.columns)} columns")
            return df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def categorize_continuous_variables(self, df):
        """
        Convert continuous variables to categorical bins
        """
        print("🔄 Categorizing continuous variables...")
        
        df_processed = df.copy()
        categorization_info = {}
        
        for column, rules in self.categorization_rules.items():
            if column not in df.columns:
                print(f"⚠️  Column {column} not found, skipping...")
                continue
            
            try:
                if rules.get('method') == 'quartiles':
                    # Use quartile-based categorization
                    quartiles = df[column].quantile([0.25, 0.5, 0.75]).values
                    bins = [-np.inf] + list(quartiles) + [np.inf]
                    df_processed[f"{rules['prefix']}_{column}"] = pd.cut(
                        df[column], 
                        bins=bins, 
                        labels=rules['labels'],
                        include_lowest=True
                    ).astype(str)
                    
                    categorization_info[column] = {
                        'method': 'quartiles',
                        'quartiles': quartiles.tolist(),
                        'labels': rules['labels']
                    }
                
                else:
                    # Use fixed bins
                    df_processed[f"{rules['prefix']}_{column}"] = pd.cut(
                        df[column], 
                        bins=rules['bins'], 
                        labels=rules['labels'],
                        include_lowest=True
                    ).astype(str)
                    
                    categorization_info[column] = {
                        'method': 'fixed_bins',
                        'bins': rules['bins'],
                        'labels': rules['labels']
                    }
                
                print(f"   ✅ Categorized {column} -> {rules['prefix']}_{column}")
                
            except Exception as e:
                print(f"   ❌ Error categorizing {column}: {e}")
        
        return df_processed, categorization_info
    
    def create_seasonal_features(self, df):
        """
        Create seasonal and temporal features
        """
        if not self.config.get('enable_seasonal_features', True):
            return df
        
        print("📅 Creating seasonal features...")
        
        df_seasonal = df.copy()
        
        try:
            # Convert dates to datetime
            df_seasonal['planting_date'] = pd.to_datetime(df_seasonal['planting_date'])
            df_seasonal['harvest_date'] = pd.to_datetime(df_seasonal['harvest_date'])
            
            # Extract planting month and season
            df_seasonal['planting_month'] = df_seasonal['planting_date'].dt.month
            
            # Define seasons
            season_mapping = {
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall'
            }
            
            df_seasonal['planting_season'] = df_seasonal['planting_month'].map(season_mapping)
            
            # Calculate growing period
            df_seasonal['growing_days'] = (
                df_seasonal['harvest_date'] - df_seasonal['planting_date']
            ).dt.days
            
            # Categorize growing period
            growing_period_bins = [0, 90, 120, 150, 365]
            growing_period_labels = ['Short', 'Medium', 'Long', 'Extended']
            df_seasonal['growing_period_category'] = pd.cut(
                df_seasonal['growing_days'],
                bins=growing_period_bins,
                labels=growing_period_labels,
                include_lowest=True
            ).astype(str)
            
            print("   ✅ Created seasonal features: planting_season, growing_period_category")
            
        except Exception as e:
            print(f"   ❌ Error creating seasonal features: {e}")
        
        return df_seasonal
    
    def create_economic_features(self, df):
        """
        Create economic performance features
        """
        if not self.config.get('enable_economic_features', True):
            return df
        
        print("💰 Creating economic features...")
        
        df_economic = df.copy()
        
        try:
            # Profit margin
            df_economic['profit_margin'] = (
                df_economic['profit_per_hectare'] / df_economic['revenue_per_hectare'] * 100
            ).fillna(0)
            
            # Categorize profit margin
            profit_margin_bins = [-100, 0, 10, 25, 100]
            profit_margin_labels = ['Loss', 'Low', 'Medium', 'High']
            df_economic['profit_margin_category'] = pd.cut(
                df_economic['profit_margin'],
                bins=profit_margin_bins,
                labels=profit_margin_labels,
                include_lowest=True
            ).astype(str)
            
            # Cost efficiency (yield per dollar spent)
            df_economic['cost_efficiency'] = (
                df_economic['yield_per_hectare'] / df_economic['cost_per_hectare']
            ).fillna(0)
            
            # Categorize cost efficiency
            efficiency_quartiles = df_economic['cost_efficiency'].quantile([0.25, 0.5, 0.75]).values
            efficiency_bins = [-np.inf] + list(efficiency_quartiles) + [np.inf]
            df_economic['cost_efficiency_category'] = pd.cut(
                df_economic['cost_efficiency'],
                bins=efficiency_bins,
                labels=['Low', 'Medium', 'High', 'VeryHigh'],
                include_lowest=True
            ).astype(str)
            
            print("   ✅ Created economic features: profit_margin_category, cost_efficiency_category")
            
        except Exception as e:
            print(f"   ❌ Error creating economic features: {e}")
        
        return df_economic
    
    def create_sustainability_features(self, df):
        """
        Create sustainability and environmental features
        """
        if not self.config.get('enable_sustainability_features', True):
            return df
        
        print("🌱 Creating sustainability features...")
        
        df_sustainability = df.copy()
        
        try:
            # Water efficiency (yield per mm of water)
            df_sustainability['water_efficiency'] = np.where(
                df_sustainability['water_usage_mm'] > 0,
                df_sustainability['yield_per_hectare'] / df_sustainability['water_usage_mm'],
                0
            )
            
            # Categorize water efficiency
            water_eff_quartiles = df_sustainability[
                df_sustainability['water_efficiency'] > 0
            ]['water_efficiency'].quantile([0.25, 0.5, 0.75]).values
            
            water_eff_bins = [0] + list(water_eff_quartiles) + [np.inf]
            df_sustainability['water_efficiency_category'] = pd.cut(
                df_sustainability['water_efficiency'],
                bins=water_eff_bins,
                labels=['None', 'Low', 'Medium', 'High'],
                include_lowest=True
            ).astype(str)
            
            # Carbon efficiency (yield per kg CO2)
            df_sustainability['carbon_efficiency'] = (
                df_sustainability['yield_per_hectare'] / df_sustainability['carbon_footprint_kg_co2']
            ).fillna(0)
            
            # Categorize carbon efficiency
            carbon_eff_quartiles = df_sustainability['carbon_efficiency'].quantile([0.25, 0.5, 0.75]).values
            carbon_eff_bins = [-np.inf] + list(carbon_eff_quartiles) + [np.inf]
            df_sustainability['carbon_efficiency_category'] = pd.cut(
                df_sustainability['carbon_efficiency'],
                bins=carbon_eff_bins,
                labels=['Low', 'Medium', 'High', 'VeryHigh'],
                include_lowest=True
            ).astype(str)
            
            # Sustainable practice indicator
            sustainability_conditions = [
                (df_sustainability['tillage_type'] == 'No_Till'),
                (df_sustainability['fertilizer_type'] == 'Organic_Compost'),
                (df_sustainability['pesticide_used'] == 'No'),
                (df_sustainability['irrigation_method'].isin(['Drip_Irrigation', 'Micro_Sprinkler']))
            ]
            
            df_sustainability['sustainable_practices_count'] = sum(sustainability_conditions)
            df_sustainability['sustainability_level'] = pd.cut(
                df_sustainability['sustainable_practices_count'],
                bins=[-1, 0, 1, 2, 4],
                labels=['Low', 'Medium', 'High', 'VeryHigh'],
                include_lowest=True
            ).astype(str)
            
            print("   ✅ Created sustainability features: water_efficiency_category, carbon_efficiency_category, sustainability_level")
            
        except Exception as e:
            print(f"   ❌ Error creating sustainability features: {e}")
        
        return df_sustainability
    
    def create_geographic_features(self, df):
        """
        Create geographic and climatic features
        """
        if not self.config.get('enable_geographic_features', True):
            return df
        
        print("🗺️ Creating geographic features...")
        
        df_geo = df.copy()
        
        try:
            # Elevation categories
            elevation_bins = [0, 200, 500, 1000, 2000]
            elevation_labels = ['Lowland', 'Rolling', 'Highland', 'Mountain']
            df_geo['elevation_category'] = pd.cut(
                df_geo['elevation'],
                bins=elevation_bins,
                labels=elevation_labels,
                include_lowest=True
            ).astype(str)
            
            # Humidity categories
            humidity_bins = [0, 50, 70, 85, 100]
            humidity_labels = ['Dry', 'Moderate', 'Humid', 'VeryHumid']
            df_geo['humidity_category'] = pd.cut(
                df_geo['humidity_percent'],
                bins=humidity_bins,
                labels=humidity_labels,
                include_lowest=True
            ).astype(str)
            
            # Solar radiation categories
            solar_bins = [0, 18, 22, 26, 35]
            solar_labels = ['Low', 'Moderate', 'High', 'VeryHigh']
            df_geo['solar_radiation_category'] = pd.cut(
                df_geo['solar_radiation'],
                bins=solar_bins,
                labels=solar_labels,
                include_lowest=True
            ).astype(str)
            
            print("   ✅ Created geographic features: elevation_category, humidity_category, solar_radiation_category")
            
        except Exception as e:
            print(f"   ❌ Error creating geographic features: {e}")
        
        return df_geo
    
    def create_transaction_format(self, df):
        """
        Convert preprocessed data to transaction format for association rules mining
        """
        print("🔄 Creating transaction format for association rules mining...")
        
        # Define columns to include in transactions
        transaction_columns = []
        
        # Add direct categorical columns
        for col in self.direct_categorical:
            if col in df.columns:
                transaction_columns.append(col)
        
        # Add categorized continuous variables
        categorized_cols = [col for col in df.columns if any(
            col.startswith(prefix) for prefix in [
                'pH_', 'OrganicMatter_', 'Nitrogen_', 'Phosphorus_', 'Potassium_',
                'Rainfall_', 'Temperature_', 'Yield_', 'Profit_', 'WaterUsage_',
                'CarbonFootprint_'
            ]
        )]
        transaction_columns.extend(categorized_cols)
        
        # Add engineered categorical features
        engineered_cols = [
            'planting_season', 'growing_period_category', 'profit_margin_category',
            'cost_efficiency_category', 'water_efficiency_category', 
            'carbon_efficiency_category', 'sustainability_level', 'elevation_category',
            'humidity_category', 'solar_radiation_category'
        ]
        
        for col in engineered_cols:
            if col in df.columns:
                transaction_columns.append(col)
        
        print(f"   📊 Selected {len(transaction_columns)} columns for transactions")
        
        # Create transactions
        transactions = []
        transaction_metadata = []
        
        for idx, row in df.iterrows():
            transaction = []
            
            for col in transaction_columns:
                if pd.notna(row[col]) and str(row[col]) != 'nan':
                    # Create item name: column_value
                    item = f"{col}_{row[col]}"
                    transaction.append(item)
            
            if len(transaction) >= 3:  # Minimum transaction size
                transactions.append(transaction)
                transaction_metadata.append({
                    'original_index': idx,
                    'farm_id': row.get('farm_id', f'FARM_{idx}'),
                    'field_id': row.get('field_id', f'FIELD_{idx}'),
                    'transaction_size': len(transaction)
                })
            
            if (idx + 1) % 1000 == 0:
                print(f"   Processed {idx + 1}/{len(df)} records...")
        
        print(f"✅ Created {len(transactions)} transactions")
        print(f"   Average transaction size: {np.mean([len(t) for t in transactions]):.1f} items")
        print(f"   Transaction size range: {min(len(t) for t in transactions)}-{max(len(t) for t in transactions)} items")
        
        return transactions, transaction_metadata, transaction_columns
    
    def save_preprocessed_data(self, df, transactions, transaction_metadata, 
                              transaction_columns, categorization_info):
        """
        Save all preprocessed data and metadata
        """
        print("💾 Saving preprocessed data...")
        
        # Create output directory
        output_dir = Path("agricultural_association_rules/data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save preprocessed dataframe
        df.to_csv(output_dir / "farm_records_preprocessed.csv", index=False)
        print(f"   📁 Saved preprocessed dataframe: {len(df)} records")
        
        # Save transactions as JSON for easy loading
        transaction_data = {
            'transactions': transactions,
            'metadata': transaction_metadata,
            'columns_used': transaction_columns,
            'statistics': {
                'total_transactions': len(transactions),
                'avg_transaction_size': np.mean([len(t) for t in transactions]),
                'min_transaction_size': min(len(t) for t in transactions),
                'max_transaction_size': max(len(t) for t in transactions),
                'unique_items': len(set(item for transaction in transactions for item in transaction))
            }
        }
        
        with open(output_dir / "agricultural_transactions.json", 'w') as f:
            json.dump(transaction_data, f, indent=2)
        print(f"   📁 Saved transaction data: {len(transactions)} transactions")
        
        # Save transactions as text file (one transaction per line)
        with open(output_dir / "agricultural_transactions.txt", 'w') as f:
            for transaction in transactions:
                f.write(','.join(transaction) + '\n')
        print(f"   📁 Saved transaction text file")
        
        # Save preprocessing metadata
        preprocessing_metadata = {
            'preprocessing_timestamp': datetime.now().isoformat(),
            'original_records': len(df),
            'transactions_created': len(transactions),
            'categorization_rules': categorization_info,
            'transaction_columns': transaction_columns,
            'config_used': self.config,
            'feature_engineering_summary': {
                'seasonal_features': self.config.get('enable_seasonal_features', True),
                'economic_features': self.config.get('enable_economic_features', True),
                'sustainability_features': self.config.get('enable_sustainability_features', True),
                'geographic_features': self.config.get('enable_geographic_features', True)
            }
        }
        
        with open(output_dir / "preprocessing_metadata.json", 'w') as f:
            json.dump(preprocessing_metadata, f, indent=2, default=str)
        print(f"   📁 Saved preprocessing metadata")
        
        return output_dir
    
    def generate_preprocessing_report(self, df_original, df_processed, transactions, 
                                    transaction_columns, categorization_info):
        """
        Generate comprehensive preprocessing report
        """
        print("\n📊 Generating Preprocessing Report...")
        
        report = {
            'preprocessing_summary': {
                'original_records': len(df_original),
                'processed_records': len(df_processed),
                'original_columns': len(df_original.columns),
                'processed_columns': len(df_processed.columns),
                'new_features_created': len(df_processed.columns) - len(df_original.columns),
                'transactions_created': len(transactions),
                'columns_used_in_transactions': len(transaction_columns)
            },
            'data_transformation': {
                'continuous_variables_categorized': len(categorization_info),
                'categorization_methods': {
                    method: sum(1 for info in categorization_info.values() 
                              if info.get('method') == method)
                    for method in ['fixed_bins', 'quartiles']
                }
            },
            'transaction_analysis': {
                'total_transactions': len(transactions),
                'avg_items_per_transaction': np.mean([len(t) for t in transactions]),
                'min_items_per_transaction': min(len(t) for t in transactions),
                'max_items_per_transaction': max(len(t) for t in transactions),
                'total_unique_items': len(set(item for transaction in transactions for item in transaction))
            },
            'feature_categories': {
                'direct_categorical': len([col for col in transaction_columns 
                                         if col in self.direct_categorical]),
                'categorized_continuous': len([col for col in transaction_columns 
                                             if any(col.startswith(prefix) for prefix in [
                                                 'pH_', 'OrganicMatter_', 'Nitrogen_', 'Phosphorus_', 
                                                 'Potassium_', 'Rainfall_', 'Temperature_', 'Yield_', 
                                                 'Profit_', 'WaterUsage_', 'CarbonFootprint_'
                                             ])]),
                'engineered_features': len([col for col in transaction_columns 
                                          if col in [
                                              'planting_season', 'growing_period_category', 
                                              'profit_margin_category', 'cost_efficiency_category',
                                              'water_efficiency_category', 'carbon_efficiency_category',
                                              'sustainability_level', 'elevation_category',
                                              'humidity_category', 'solar_radiation_category'
                                          ]])
            }
        }
        
        # Display report
        print("\n" + "=" * 60)
        print("📈 PREPROCESSING REPORT")
        print("=" * 60)
        print(f"📊 Original dataset: {report['preprocessing_summary']['original_records']:,} records, "
              f"{report['preprocessing_summary']['original_columns']} columns")
        print(f"📊 Processed dataset: {report['preprocessing_summary']['processed_records']:,} records, "
              f"{report['preprocessing_summary']['processed_columns']} columns")
        print(f"✨ New features created: {report['preprocessing_summary']['new_features_created']}")
        print(f"🔄 Transactions created: {report['preprocessing_summary']['transactions_created']:,}")
        print(f"📋 Average items per transaction: {report['transaction_analysis']['avg_items_per_transaction']:.1f}")
        print(f"🎯 Unique items for mining: {report['transaction_analysis']['total_unique_items']:,}")
        
        print(f"\n📑 Feature Categories:")
        print(f"   Direct categorical: {report['feature_categories']['direct_categorical']}")
        print(f"   Categorized continuous: {report['feature_categories']['categorized_continuous']}")
        print(f"   Engineered features: {report['feature_categories']['engineered_features']}")
        
        return report

def main():
    """
    Main preprocessing function
    """
    print("🔄 Agricultural Data Preprocessing - Step 3")
    print("=" * 60)
    print("Converting raw agricultural data to association rules format")
    print("=" * 60)
    
    try:
        # Initialize preprocessor
        preprocessor = AgriculturalDataPreprocessor()
        
        # Load raw data
        raw_data_path = "agricultural_association_rules/data/raw/farm_records.csv"
        df_original = preprocessor.load_agricultural_data(raw_data_path)
        
        # Apply all preprocessing steps
        print("\n🔧 Applying preprocessing steps...")
        
        # Step 1: Categorize continuous variables
        df_processed, categorization_info = preprocessor.categorize_continuous_variables(df_original)
        
        # Step 2: Create seasonal features
        df_processed = preprocessor.create_seasonal_features(df_processed)
        
        # Step 3: Create economic features
        df_processed = preprocessor.create_economic_features(df_processed)
        
        # Step 4: Create sustainability features
        df_processed = preprocessor.create_sustainability_features(df_processed)
        
        # Step 5: Create geographic features
        df_processed = preprocessor.create_geographic_features(df_processed)
        
        # Step 6: Create transaction format
        transactions, transaction_metadata, transaction_columns = preprocessor.create_transaction_format(df_processed)
        
        # Step 7: Save all processed data
        output_dir = preprocessor.save_preprocessed_data(
            df_processed, transactions, transaction_metadata, 
            transaction_columns, categorization_info
        )
        
        # Step 8: Generate comprehensive report
        report = preprocessor.generate_preprocessing_report(
            df_original, df_processed, transactions, 
            transaction_columns, categorization_info
        )
        
        # Save report
        with open(output_dir / "preprocessing_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n🎉 Step 3 Complete!")
        print(f"📁 All files saved to: {output_dir}")
        print("🎯 Ready for Step 4: Association Rules Mining")
        print("=" * 60)
        
        return df_processed, transactions, report
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    # Run preprocessing
    processed_data, transactions, report = main()