
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Association Rules Mining
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️  Plotly not available, using matplotlib for visualizations")

class AgriculturalAssociationRulesMiner:
    """
    Mine association rules from agricultural transaction data
    """
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.frequent_itemsets = None
        self.association_rules = None
        self.transaction_data = None
        self.df_encoded = None
        
    def _default_config(self):
        """Default configuration for association rules mining"""
        return {
            'min_support': 0.05,      # 5% minimum support
            'min_confidence': 0.6,    # 60% minimum confidence
            'min_lift': 1.2,          # 20% lift improvement
            'max_len': 5,             # Maximum itemset length
            'metric': 'confidence',   # Primary metric for rule generation
            'top_rules_count': 50     # Number of top rules to analyze
        }
    
    def load_transaction_data(self, data_path):
        """
        Load preprocessed transaction data
        """
        print(f"📂 Loading transaction data from {data_path}")
        
        try:
            with open(data_path, 'r') as f:
                self.transaction_data = json.load(f)
            
            transactions = self.transaction_data['transactions']
            metadata = self.transaction_data['metadata']
            statistics = self.transaction_data['statistics']
            
            print(f"✅ Loaded {len(transactions)} transactions")
            print(f"   📊 Average transaction size: {statistics['avg_transaction_size']:.1f} items")
            print(f"   🎯 Unique items: {statistics['unique_items']}")
            print(f"   📏 Transaction size range: {statistics['min_transaction_size']}-{statistics['max_transaction_size']}")
            
            return transactions, metadata, statistics
            
        except Exception as e:
            print(f"❌ Error loading transaction data: {e}")
            raise
    
    def encode_transactions(self, transactions):
        """
        Encode transactions for mlxtend association rules mining
        """
        print("🔄 Encoding transactions for association rules mining...")
        
        try:
            # Use TransactionEncoder from mlxtend
            te = TransactionEncoder()
            te_array = te.fit_transform(transactions)
            self.df_encoded = pd.DataFrame(te_array, columns=te.columns_)
            
            print(f"✅ Encoded {len(self.df_encoded)} transactions")
            print(f"   📊 Items (columns): {len(self.df_encoded.columns)}")
            print(f"   💾 Memory usage: {self.df_encoded.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
            
            return self.df_encoded
            
        except Exception as e:
            print(f"❌ Error encoding transactions: {e}")
            raise
    
    def mine_frequent_itemsets(self, df_encoded):
        """
        Mine frequent itemsets using Apriori algorithm
        """
        print(f"⛏️  Mining frequent itemsets (min_support={self.config['min_support']})...")
        
        try:
            self.frequent_itemsets = apriori(
                df_encoded,
                min_support=self.config['min_support'],
                use_colnames=True,
                max_len=self.config['max_len'],
                verbose=1
            )
            
            if len(self.frequent_itemsets) == 0:
                print("⚠️  No frequent itemsets found! Consider lowering min_support.")
                return None
            
            print(f"✅ Found {len(self.frequent_itemsets)} frequent itemsets")
            
            # Analyze itemset distribution by length
            itemset_lengths = self.frequent_itemsets['itemsets'].apply(len)
            length_distribution = itemset_lengths.value_counts().sort_index()
            
            print("   📊 Itemset length distribution:")
            for length, count in length_distribution.items():
                print(f"      Length {length}: {count} itemsets")
            
            return self.frequent_itemsets
            
        except Exception as e:
            print(f"❌ Error mining frequent itemsets: {e}")
            raise
    
    def generate_association_rules(self, frequent_itemsets):
        """
        Generate association rules from frequent itemsets
        """
        print(f"🔗 Generating association rules (min_confidence={self.config['min_confidence']})...")
        
        try:
            self.association_rules = association_rules(
                frequent_itemsets,
                metric=self.config['metric'],
                min_threshold=self.config['min_confidence'],
                num_itemsets=len(frequent_itemsets)
            )
            
            if len(self.association_rules) == 0:
                print("⚠️  No association rules found! Consider lowering min_confidence.")
                return None
            
            # Filter by lift
            rules_before_lift = len(self.association_rules)
            self.association_rules = self.association_rules[
                self.association_rules['lift'] >= self.config['min_lift']
            ]
            
            print(f"✅ Generated {len(self.association_rules)} association rules")
            print(f"   🔍 Rules before lift filter: {rules_before_lift}")
            print(f"   🎯 Rules after lift filter (≥{self.config['min_lift']}): {len(self.association_rules)}")
            
            if len(self.association_rules) > 0:
                # Sort by lift (descending)
                self.association_rules = self.association_rules.sort_values('lift', ascending=False)
                
                # Add rule interpretations
                self.association_rules = self._interpret_agricultural_rules(self.association_rules)
                
                # Display top rules summary
                print(f"\n🏆 Top 5 Association Rules by Lift:")
                top_rules = self.association_rules.head(5)
                for idx, rule in top_rules.iterrows():
                    antecedent = ', '.join(list(rule['antecedents']))
                    consequent = ', '.join(list(rule['consequents']))
                    print(f"   {antecedent[:50]}... → {consequent[:50]}...")
                    print(f"      Support: {rule['support']:.3f}, Confidence: {rule['confidence']:.3f}, Lift: {rule['lift']:.2f}")
            
            return self.association_rules
            
        except Exception as e:
            print(f"❌ Error generating association rules: {e}")
            raise
    
    def _interpret_agricultural_rules(self, rules):
        """
        Add agricultural interpretation to association rules
        """
        print("🧠 Adding agricultural interpretations to rules...")
        
        interpretations = []
        categories = []
        
        for idx, rule in rules.iterrows():
            antecedent_items = list(rule['antecedents'])
            consequent_items = list(rule['consequents'])
            all_items = antecedent_items + consequent_items
            
            # Determine rule category and interpretation
            category, interpretation = self._categorize_agricultural_rule(all_items, rule)
            
            categories.append(category)
            interpretations.append(interpretation)
        
        rules['rule_category'] = categories
        rules['agricultural_interpretation'] = interpretations
        
        return rules
    
    def _categorize_agricultural_rule(self, items, rule):
        """
        Categorize and interpret agricultural rules
        """
        items_str = ' '.join(items).lower()
        
        # High-impact rules
        if rule['lift'] > 3.0 and rule['confidence'] > 0.8:
            if 'yield_' in items_str and ('high' in items_str or 'veryhigh' in items_str):
                return "High-Impact Yield", "Strong predictor of high crop yields"
            elif 'profit_' in items_str and ('high' in items_str or 'veryhigh' in items_str):
                return "High-Impact Profit", "Strong predictor of high profitability"
        
        # Soil management rules
        if any(term in items_str for term in ['soil_type', 'ph_', 'organicmatter', 'nitrogen', 'phosphorus', 'potassium']):
            if 'yield_' in items_str:
                return "Soil-Yield", "Soil conditions affecting crop yield"
            elif 'crop_type' in items_str:
                return "Soil-Crop", "Optimal soil conditions for specific crops"
            else:
                return "Soil Management", "Soil nutrition and management practices"
        
        # Climate and environmental rules
        if any(term in items_str for term in ['rainfall', 'temperature', 'humidity', 'solar']):
            if 'yield_' in items_str:
                return "Climate-Yield", "Weather conditions affecting productivity"
            elif 'crop_type' in items_str:
                return "Climate-Crop", "Climate suitability for crops"
            else:
                return "Environmental", "Climate and environmental factors"
        
        # Farming practice rules
        if any(term in items_str for term in ['fertilizer', 'irrigation', 'tillage']):
            if 'sustainability' in items_str:
                return "Sustainable Practices", "Environmentally friendly farming methods"
            elif 'yield_' in items_str or 'profit_' in items_str:
                return "Practice-Performance", "Farming practices affecting outcomes"
            else:
                return "Farm Management", "Agricultural management practices"
        
        # Economic rules
        if any(term in items_str for term in ['profit', 'cost_efficiency', 'profit_margin']):
            return "Economic Performance", "Financial performance factors"
        
        # Sustainability rules
        if any(term in items_str for term in ['water_efficiency', 'carbon', 'sustainability_level']):
            return "Sustainability", "Environmental sustainability metrics"
        
        # Regional rules
        if 'region_' in items_str:
            return "Regional Patterns", "Geographic and regional farming patterns"
        
        # Default
        return "General Agriculture", "General agricultural associations"
    
    def analyze_rule_patterns(self):
        """
        Analyze patterns in discovered association rules
        """
        if self.association_rules is None or len(self.association_rules) == 0:
            print("⚠️  No association rules to analyze")
            return None
        
        print("\n📊 Analyzing Association Rule Patterns...")
        
        analysis = {}
        
        # Rule category distribution
        category_dist = self.association_rules['rule_category'].value_counts()
        analysis['category_distribution'] = category_dist.to_dict()
        
        print(f"🏷️  Rule Categories:")
        for category, count in category_dist.items():
            percentage = (count / len(self.association_rules)) * 100
            print(f"   {category}: {count} rules ({percentage:.1f}%)")
        
        # Performance metrics analysis
        metrics_stats = {
            'support': {
                'mean': self.association_rules['support'].mean(),
                'median': self.association_rules['support'].median(),
                'std': self.association_rules['support'].std(),
                'min': self.association_rules['support'].min(),
                'max': self.association_rules['support'].max()
            },
            'confidence': {
                'mean': self.association_rules['confidence'].mean(),
                'median': self.association_rules['confidence'].median(),
                'std': self.association_rules['confidence'].std(),
                'min': self.association_rules['confidence'].min(),
                'max': self.association_rules['confidence'].max()
            },
            'lift': {
                'mean': self.association_rules['lift'].mean(),
                'median': self.association_rules['lift'].median(),
                'std': self.association_rules['lift'].std(),
                'min': self.association_rules['lift'].min(),
                'max': self.association_rules['lift'].max()
            }
        }
        analysis['metrics_statistics'] = metrics_stats
        
        print(f"\n📈 Rule Performance Metrics:")
        for metric, stats in metrics_stats.items():
            print(f"   {metric.capitalize()}:")
            print(f"      Mean: {stats['mean']:.3f}, Median: {stats['median']:.3f}")
            print(f"      Range: {stats['min']:.3f} - {stats['max']:.3f}")
        
        # Top rules by category
        analysis['top_rules_by_category'] = {}
        for category in category_dist.index[:5]:  # Top 5 categories
            category_rules = self.association_rules[
                self.association_rules['rule_category'] == category
            ].head(3)  # Top 3 rules per category
            
            analysis['top_rules_by_category'][category] = []
            for idx, rule in category_rules.iterrows():
                rule_info = {
                    'antecedents': list(rule['antecedents']),
                    'consequents': list(rule['consequents']),
                    'support': rule['support'],
                    'confidence': rule['confidence'],
                    'lift': rule['lift'],
                    'interpretation': rule['agricultural_interpretation']
                }
                analysis['top_rules_by_category'][category].append(rule_info)
        
        return analysis
    
    def save_results(self, output_dir=None):
        """
        Save association rules mining results
        """
        if output_dir is None:
            output_dir = Path("agricultural_association_rules/results/models")
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Saving association rules results to {output_dir}...")
        
        # Save frequent itemsets
        if self.frequent_itemsets is not None:
            # Convert frozenset to string for JSON serialization
            frequent_itemsets_json = self.frequent_itemsets.copy()
            frequent_itemsets_json['itemsets'] = frequent_itemsets_json['itemsets'].apply(
                lambda x: list(x)
            )
            frequent_itemsets_json.to_csv(output_dir / "frequent_itemsets.csv", index=False)
            print(f"   📁 Saved frequent_itemsets.csv ({len(self.frequent_itemsets)} itemsets)")
        
        # Save association rules
        if self.association_rules is not None:
            # Convert frozenset to string for CSV
            rules_csv = self.association_rules.copy()
            rules_csv['antecedents'] = rules_csv['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules_csv['consequents'] = rules_csv['consequents'].apply(lambda x: ', '.join(list(x)))
            rules_csv.to_csv(output_dir / "association_rules.csv", index=False)
            print(f"   📁 Saved association_rules.csv ({len(self.association_rules)} rules)")
            
            # Save top rules summary
            top_rules = self.association_rules.head(self.config['top_rules_count'])
            top_rules_summary = []
            
            for idx, rule in top_rules.iterrows():
                summary = {
                    'rule_id': idx,
                    'antecedents': list(rule['antecedents']),
                    'consequents': list(rule['consequents']),
                    'support': float(rule['support']),
                    'confidence': float(rule['confidence']),
                    'lift': float(rule['lift']),
                    'conviction': float(rule.get('conviction', 0)),
                    'rule_category': rule['rule_category'],
                    'agricultural_interpretation': rule['agricultural_interpretation']
                }
                top_rules_summary.append(summary)
            
            with open(output_dir / "top_association_rules.json", 'w') as f:
                json.dump(top_rules_summary, f, indent=2)
            print(f"   📁 Saved top_association_rules.json ({len(top_rules_summary)} rules)")
        
        # Save analysis results
        analysis_results = self.analyze_rule_patterns()
        if analysis_results:
            with open(output_dir / "rule_analysis.json", 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            print(f"   📁 Saved rule_analysis.json")
        
        # Save mining configuration and metadata
        mining_metadata = {
            'mining_timestamp': datetime.now().isoformat(),
            'configuration': self.config,
            'results_summary': {
                'frequent_itemsets_count': len(self.frequent_itemsets) if self.frequent_itemsets is not None else 0,
                'association_rules_count': len(self.association_rules) if self.association_rules is not None else 0,
                'transaction_count': len(self.transaction_data['transactions']) if self.transaction_data else 0
            }
        }
        
        with open(output_dir / "mining_metadata.json", 'w') as f:
            json.dump(mining_metadata, f, indent=2)
        print(f"   📁 Saved mining_metadata.json")
        
        return output_dir
    
    def create_visualizations(self, output_dir=None):
        """
        Create visualizations for association rules
        """
        if self.association_rules is None or len(self.association_rules) == 0:
            print("⚠️  No association rules to visualize")
            return
        
        if output_dir is None:
            output_dir = Path("agricultural_association_rules/results/figures")
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📈 Creating association rules visualizations...")
        
        # Set up matplotlib style
        plt.style.use('default')
        
        # 1. Rules scatter plot: Support vs Confidence (colored by Lift)
        self._plot_support_confidence_lift(output_dir)
        
        # 2. Rule category distribution
        self._plot_rule_categories(output_dir)
        
        # 3. Top rules by lift
        self._plot_top_rules(output_dir)
        
        # 4. Metrics distribution
        self._plot_metrics_distribution(output_dir)
        
        print(f"✅ Visualizations saved to {output_dir}")
    
    def _plot_support_confidence_lift(self, output_dir):
        """Plot support vs confidence scatter plot colored by lift"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        scatter = ax.scatter(
            self.association_rules['support'],
            self.association_rules['confidence'],
            c=self.association_rules['lift'],
            cmap='viridis',
            alpha=0.7,
            s=50
        )
        
        ax.set_xlabel('Support', fontsize=12)
        ax.set_ylabel('Confidence', fontsize=12)
        ax.set_title('Association Rules: Support vs Confidence (colored by Lift)', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Lift', fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "support_confidence_lift.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   📊 Saved support_confidence_lift.png")
    
    def _plot_rule_categories(self, output_dir):
        """Plot rule category distribution"""
        category_counts = self.association_rules['rule_category'].value_counts()
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        bars = ax.bar(range(len(category_counts)), category_counts.values, 
                     color='skyblue', alpha=0.8)
        
        ax.set_xlabel('Rule Categories', fontsize=12)
        ax.set_ylabel('Number of Rules', fontsize=12)
        ax.set_title('Distribution of Association Rule Categories', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(category_counts)))
        ax.set_xticklabels(category_counts.index, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / "rule_categories.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   📊 Saved rule_categories.png")
    
    def _plot_top_rules(self, output_dir):
        """Plot top rules by lift"""
        top_rules = self.association_rules.head(15)
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # Create rule labels (shortened)
        rule_labels = []
        for idx, rule in top_rules.iterrows():
            antecedent = ', '.join(list(rule['antecedents']))[:30]
            consequent = ', '.join(list(rule['consequents']))[:20]
            rule_labels.append(f"{antecedent}... → {consequent}...")
        
        y_pos = np.arange(len(rule_labels))
        
        bars = ax.barh(y_pos, top_rules['lift'], color='lightcoral', alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(rule_labels, fontsize=9)
        ax.set_xlabel('Lift', fontsize=12)
        ax.set_title('Top 15 Association Rules by Lift', fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{width:.2f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_dir / "top_rules_by_lift.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   📊 Saved top_rules_by_lift.png")
    
    def _plot_metrics_distribution(self, output_dir):
        """Plot distribution of metrics"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Support distribution
        axes[0].hist(self.association_rules['support'], bins=30, alpha=0.7, color='skyblue')
        axes[0].set_xlabel('Support')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Support Distribution')
        axes[0].grid(True, alpha=0.3)
        
        # Confidence distribution
        axes[1].hist(self.association_rules['confidence'], bins=30, alpha=0.7, color='lightgreen')
        axes[1].set_xlabel('Confidence')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Confidence Distribution')
        axes[1].grid(True, alpha=0.3)
        
        # Lift distribution
        axes[2].hist(self.association_rules['lift'], bins=30, alpha=0.7, color='lightcoral')
        axes[2].set_xlabel('Lift')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Lift Distribution')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "metrics_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   📊 Saved metrics_distribution.png")

def main():
    """
    Main association rules mining function
    """
    print("⛏️  Agricultural Association Rules Mining - Step 4")
    print("=" * 60)
    print("Discovering patterns in agricultural practices and outcomes")
    print("=" * 60)
    
    try:
        # Initialize miner with configuration
        miner = AgriculturalAssociationRulesMiner()
        
        print(f"🔧 Mining Configuration:")
        print(f"   Minimum Support: {miner.config['min_support']}")
        print(f"   Minimum Confidence: {miner.config['min_confidence']}")
        print(f"   Minimum Lift: {miner.config['min_lift']}")
        print(f"   Maximum Itemset Length: {miner.config['max_len']}")
        
        # Load transaction data
        transaction_file = "agricultural_association_rules/data/processed/agricultural_transactions.json"
        transactions, metadata, statistics = miner.load_transaction_data(transaction_file)
        
        # Encode transactions
        df_encoded = miner.encode_transactions(transactions)
        
        # Mine frequent itemsets
        frequent_itemsets = miner.mine_frequent_itemsets(df_encoded)
        
        if frequent_itemsets is not None and len(frequent_itemsets) > 0:
            # Generate association rules
            association_rules = miner.generate_association_rules(frequent_itemsets)
            
            if association_rules is not None and len(association_rules) > 0:
                # Analyze patterns
                analysis = miner.analyze_rule_patterns()
                
                # Save results
                output_dir = miner.save_results()
                
                # Create visualizations
                miner.create_visualizations()
                
                print(f"\n🎉 Step 4 Complete!")
                print(f"📁 Results saved to: {output_dir}")
                print(f"📊 Found {len(frequent_itemsets)} frequent itemsets")
                print(f"🔗 Generated {len(association_rules)} association rules")
                print("🎯 Ready for Step 5: Analysis and Interpretation")
                print("=" * 60)
                
                return miner, analysis
            else:
                print("❌ No association rules generated. Consider adjusting parameters.")
                return None, None
        else:
            print("❌ No frequent itemsets found. Consider lowering min_support.")
            return None, None
        
    except Exception as e:
        print(f"❌ Association rules mining failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    # Run association rules mining
    miner, analysis = main()