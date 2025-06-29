import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

class AgriculturalInsightsAnalyzer:
    """
    Analyze association rules results and generate agricultural insights
    """
    
    def __init__(self, results_dir=None):
        self.results_dir = Path(results_dir) if results_dir else Path("agricultural_association_rules/results")
        self.association_rules = None
        self.frequent_itemsets = None
        self.rule_analysis = None
        self.agricultural_insights = {}
        
    def load_mining_results(self):
        """
        Load association rules mining results
        """
        print("📂 Loading association rules mining results...")
        
        try:
            # Load association rules
            rules_file = self.results_dir / "models" / "association_rules.csv"
            if rules_file.exists():
                self.association_rules = pd.read_csv(rules_file)
                print(f"✅ Loaded {len(self.association_rules)} association rules")
            else:
                print("⚠️  Association rules file not found")
                return False
            
            # Load frequent itemsets
            itemsets_file = self.results_dir / "models" / "frequent_itemsets.csv"
            if itemsets_file.exists():
                self.frequent_itemsets = pd.read_csv(itemsets_file)
                print(f"✅ Loaded {len(self.frequent_itemsets)} frequent itemsets")
            
            # Load rule analysis
            analysis_file = self.results_dir / "models" / "rule_analysis.json"
            if analysis_file.exists():
                with open(analysis_file, 'r') as f:
                    self.rule_analysis = json.load(f)
                print(f"✅ Loaded rule analysis data")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading results: {e}")
            return False
    
    def analyze_high_yield_patterns(self):
        """
        Analyze patterns that lead to high crop yields
        """
        print("\n🌾 Analyzing High-Yield Patterns...")
        
        # Filter rules that lead to high yield
        yield_rules = self.association_rules[
            self.association_rules['consequents'].str.contains('Yield_.*High', na=False, regex=True)
        ].copy()
        
        if len(yield_rules) == 0:
            print("⚠️  No high-yield rules found")
            return {}
        
        # Sort by lift and confidence
        yield_rules = yield_rules.sort_values(['lift', 'confidence'], ascending=False)
        
        print(f"📊 Found {len(yield_rules)} rules leading to high yields")
        
        # Analyze top yield patterns
        top_yield_patterns = []
        for idx, rule in yield_rules.head(10).iterrows():
            pattern = {
                'conditions': rule['antecedents'],
                'yield_outcome': rule['consequents'],
                'support': rule['support'],
                'confidence': rule['confidence'],
                'lift': rule['lift'],
                'practical_meaning': self._interpret_yield_pattern(rule['antecedents'])
            }
            top_yield_patterns.append(pattern)
        
        # Identify most common factors in high-yield rules
        all_conditions = []
        for rule in yield_rules['antecedents']:
            conditions = [cond.strip() for cond in rule.split(',')]
            all_conditions.extend(conditions)
        
        condition_frequency = pd.Series(all_conditions).value_counts()
        
        yield_insights = {
            'total_yield_rules': len(yield_rules),
            'top_patterns': top_yield_patterns,
            'most_important_factors': condition_frequency.head(10).to_dict(),
            'average_confidence': yield_rules['confidence'].mean(),
            'average_lift': yield_rules['lift'].mean()
        }
        
        # Display insights
        print(f"\n🏆 Top High-Yield Success Factors:")
        for factor, count in condition_frequency.head(5).items():
            percentage = (count / len(yield_rules)) * 100
            print(f"   {factor}: appears in {count} rules ({percentage:.1f}%)")
        
        print(f"\n📈 High-Yield Rule Statistics:")
        print(f"   Average Confidence: {yield_rules['confidence'].mean():.3f}")
        print(f"   Average Lift: {yield_rules['lift'].mean():.2f}")
        print(f"   Strongest Rule Lift: {yield_rules['lift'].max():.2f}")
        
        self.agricultural_insights['high_yield_patterns'] = yield_insights
        return yield_insights
    
    def analyze_profitability_patterns(self):
        """
        Analyze patterns that lead to high profitability
        """
        print("\n💰 Analyzing Profitability Patterns...")
        
        # Filter rules that lead to high profit
        profit_rules = self.association_rules[
            self.association_rules['consequents'].str.contains('Profit_.*High|profit_margin_category_High', na=False, regex=True)
        ].copy()
        
        if len(profit_rules) == 0:
            print("⚠️  No high-profit rules found")
            return {}
        
        profit_rules = profit_rules.sort_values(['lift', 'confidence'], ascending=False)
        
        print(f"📊 Found {len(profit_rules)} rules leading to high profitability")
        
        # Analyze cost-effectiveness patterns
        cost_efficiency_rules = self.association_rules[
            self.association_rules['consequents'].str.contains('cost_efficiency_category_High', na=False, regex=True)
        ].copy()
        
        profitability_insights = {
            'total_profit_rules': len(profit_rules),
            'cost_efficiency_rules': len(cost_efficiency_rules),
            'top_profit_strategies': self._extract_profit_strategies(profit_rules),
            'cost_optimization_factors': self._extract_cost_factors(cost_efficiency_rules),
            'average_profit_confidence': profit_rules['confidence'].mean(),
            'average_profit_lift': profit_rules['lift'].mean()
        }
        
        # Display insights
        print(f"\n💡 Top Profitability Strategies:")
        for i, strategy in enumerate(profitability_insights['top_profit_strategies'][:3], 1):
            print(f"   {i}. {strategy['description']}")
            print(f"      Confidence: {strategy['confidence']:.3f}, Lift: {strategy['lift']:.2f}")
        
        self.agricultural_insights['profitability_patterns'] = profitability_insights
        return profitability_insights
    
    def analyze_sustainability_patterns(self):
        """
        Analyze patterns that lead to sustainable farming
        """
        print("\n🌱 Analyzing Sustainability Patterns...")
        
        # Filter rules related to sustainability
        sustainability_rules = self.association_rules[
            self.association_rules['consequents'].str.contains(
                'sustainability_level_High|water_efficiency_category_High|carbon_efficiency_category_High', 
                na=False, regex=True
            )
        ].copy()
        
        if len(sustainability_rules) == 0:
            print("⚠️  No sustainability rules found")
            return {}
        
        sustainability_rules = sustainability_rules.sort_values(['lift', 'confidence'], ascending=False)
        
        print(f"📊 Found {len(sustainability_rules)} sustainability-related rules")
        
        # Analyze sustainable practices
        sustainable_practices = []
        for idx, rule in sustainability_rules.head(10).iterrows():
            practice = {
                'practices': rule['antecedents'],
                'sustainability_outcome': rule['consequents'],
                'confidence': rule['confidence'],
                'lift': rule['lift'],
                'environmental_benefit': self._interpret_sustainability_benefit(rule)
            }
            sustainable_practices.append(practice)
        
        # Identify eco-friendly farming combinations
        eco_friendly_combinations = self._identify_eco_combinations(sustainability_rules)
        
        sustainability_insights = {
            'total_sustainability_rules': len(sustainability_rules),
            'sustainable_practices': sustainable_practices,
            'eco_friendly_combinations': eco_friendly_combinations,
            'average_sustainability_confidence': sustainability_rules['confidence'].mean(),
            'average_sustainability_lift': sustainability_rules['lift'].mean()
        }
        
        # Display insights
        print(f"\n🌿 Top Sustainable Farming Practices:")
        for i, combo in enumerate(eco_friendly_combinations[:3], 1):
            print(f"   {i}. {combo['description']}")
            print(f"      Environmental Impact: {combo['impact']}")
            print(f"      Confidence: {combo['confidence']:.3f}")
        
        self.agricultural_insights['sustainability_patterns'] = sustainability_insights
        return sustainability_insights
    
    def analyze_crop_soil_combinations(self):
        """
        Analyze optimal crop-soil combinations
        """
        print("\n🌍 Analyzing Crop-Soil Combinations...")
        
        # Filter rules with crop and soil relationships
        crop_soil_rules = self.association_rules[
            (self.association_rules['antecedents'].str.contains('crop_type_', na=False)) &
            (self.association_rules['antecedents'].str.contains('soil_type_|pH_', na=False))
        ].copy()
        
        if len(crop_soil_rules) == 0:
            print("⚠️  No crop-soil combination rules found")
            return {}
        
        crop_soil_rules = crop_soil_rules.sort_values(['lift', 'confidence'], ascending=False)
        
        print(f"📊 Found {len(crop_soil_rules)} crop-soil combination rules")
        
        # Extract crop-specific recommendations
        crop_recommendations = {}
        for idx, rule in crop_soil_rules.iterrows():
            crops = self._extract_crops_from_rule(rule['antecedents'])
            soil_conditions = self._extract_soil_conditions_from_rule(rule['antecedents'])
            
            for crop in crops:
                if crop not in crop_recommendations:
                    crop_recommendations[crop] = []
                
                recommendation = {
                    'soil_conditions': soil_conditions,
                    'confidence': rule['confidence'],
                    'lift': rule['lift'],
                    'outcome': rule['consequents']
                }
                crop_recommendations[crop].append(recommendation)
        
        # Sort recommendations by confidence for each crop
        for crop in crop_recommendations:
            crop_recommendations[crop] = sorted(
                crop_recommendations[crop], 
                key=lambda x: x['confidence'], 
                reverse=True
            )[:5]  # Top 5 recommendations per crop
        
        crop_soil_insights = {
            'total_combinations': len(crop_soil_rules),
            'crop_specific_recommendations': crop_recommendations,
            'most_versatile_soils': self._find_versatile_soils(crop_soil_rules),
            'most_adaptable_crops': self._find_adaptable_crops(crop_soil_rules)
        }
        
        # Display insights
        print(f"\n🌾 Crop-Specific Soil Recommendations:")
        for crop, recommendations in list(crop_recommendations.items())[:3]:
            print(f"   {crop}:")
            for rec in recommendations[:2]:
                print(f"      - {rec['soil_conditions']} (Confidence: {rec['confidence']:.3f})")
        
        self.agricultural_insights['crop_soil_combinations'] = crop_soil_insights
        return crop_soil_insights
    
    def analyze_regional_patterns(self):
        """
        Analyze regional farming patterns
        """
        print("\n🗺️ Analyzing Regional Patterns...")
        
        # Filter rules with regional information
        regional_rules = self.association_rules[
            self.association_rules['antecedents'].str.contains('region_', na=False) |
            self.association_rules['consequents'].str.contains('region_', na=False)
        ].copy()
        
        if len(regional_rules) == 0:
            print("⚠️  No regional pattern rules found")
            return {}
        
        regional_rules = regional_rules.sort_values(['lift', 'confidence'], ascending=False)
        
        print(f"📊 Found {len(regional_rules)} regional pattern rules")
        
        # Extract region-specific insights
        regional_insights = {}
        regions = self._extract_regions_from_rules(regional_rules)
        
        for region in regions:
            region_rules = regional_rules[
                regional_rules['antecedents'].str.contains(f'region_{region}', na=False) |
                regional_rules['consequents'].str.contains(f'region_{region}', na=False)
            ]
            
            regional_insights[region] = {
                'total_rules': len(region_rules),
                'characteristic_practices': self._extract_regional_practices(region_rules, region),
                'performance_indicators': self._extract_regional_performance(region_rules),
                'climate_adaptations': self._extract_climate_adaptations(region_rules)
            }
        
        # Display insights
        print(f"\n🌎 Regional Farming Characteristics:")
        for region, insights in list(regional_insights.items())[:3]:
            print(f"   {region}:")
            print(f"      Rules: {insights['total_rules']}")
            if insights['characteristic_practices']:
                print(f"      Key Practice: {insights['characteristic_practices'][0]}")
        
        self.agricultural_insights['regional_patterns'] = regional_insights
        return regional_insights
    
    def generate_recommendations(self):
        """
        Generate actionable agricultural recommendations
        """
        print("\n📋 Generating Agricultural Recommendations...")
        
        recommendations = {
            'yield_optimization': self._generate_yield_recommendations(),
            'profit_maximization': self._generate_profit_recommendations(),
            'sustainability_improvement': self._generate_sustainability_recommendations(),
            'crop_selection_guide': self._generate_crop_selection_guide(),
            'regional_best_practices': self._generate_regional_recommendations()
        }
        
        # Display key recommendations
        print(f"\n🎯 Key Agricultural Recommendations:")
        print(f"\n🌾 Yield Optimization:")
        for rec in recommendations['yield_optimization'][:3]:
            print(f"   • {rec}")
        
        print(f"\n💰 Profit Maximization:")
        for rec in recommendations['profit_maximization'][:3]:
            print(f"   • {rec}")
        
        print(f"\n🌱 Sustainability:")
        for rec in recommendations['sustainability_improvement'][:3]:
            print(f"   • {rec}")
        
        self.agricultural_insights['recommendations'] = recommendations
        return recommendations
    
    def create_insights_dashboard(self):
        """
        Create comprehensive insights dashboard
        """
        print("\n📊 Creating Agricultural Insights Dashboard...")
        
        if not PLOTLY_AVAILABLE:
            print("⚠️  Plotly not available, creating matplotlib dashboard")
            return self._create_matplotlib_dashboard()
        
        # Create interactive Plotly dashboard
        dashboard_data = self._prepare_dashboard_data()
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Yield Success Factors', 'Profitability Patterns',
                'Sustainability Metrics', 'Crop-Soil Combinations',
                'Regional Performance', 'Rule Category Distribution'
            ),
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "heatmap"}],
                [{"type": "bar"}, {"type": "pie"}]
            ]
        )
        
        # Add plots to dashboard
        self._add_yield_factors_plot(fig, dashboard_data, 1, 1)
        self._add_profitability_plot(fig, dashboard_data, 1, 2)
        self._add_sustainability_plot(fig, dashboard_data, 2, 1)
        self._add_crop_soil_heatmap(fig, dashboard_data, 2, 2)
        self._add_regional_performance(fig, dashboard_data, 3, 1)
        self._add_rule_categories_pie(fig, dashboard_data, 3, 2)
        
        # Update layout
        fig.update_layout(
            title="Agricultural Association Rules - Insights Dashboard",
            height=1200,
            showlegend=False
        )
        
        # Save dashboard
        output_dir = self.results_dir / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if PLOTLY_AVAILABLE:
            fig.write_html(output_dir / "agricultural_insights_dashboard.html")
            print(f"✅ Interactive dashboard saved: agricultural_insights_dashboard.html")
        
        return fig
    
    def save_insights_report(self):
        """
        Save comprehensive insights report
        """
        print("\n💾 Saving Agricultural Insights Report...")
        
        # Create comprehensive report
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_rules_analyzed': len(self.association_rules) if self.association_rules is not None else 0,
                'high_yield_rules': len(self.agricultural_insights.get('high_yield_patterns', {}).get('top_patterns', [])),
                'profitability_rules': self.agricultural_insights.get('profitability_patterns', {}).get('total_profit_rules', 0),
                'sustainability_rules': self.agricultural_insights.get('sustainability_patterns', {}).get('total_sustainability_rules', 0),
                'crop_soil_combinations': self.agricultural_insights.get('crop_soil_combinations', {}).get('total_combinations', 0)
            },
            'key_insights': self.agricultural_insights,
            'actionable_recommendations': self.agricultural_insights.get('recommendations', {}),
            'methodology': {
                'association_rules_mining': 'Apriori algorithm with agricultural transaction data',
                'confidence_threshold': 'Rules with confidence >= 60%',
                'lift_threshold': 'Rules with lift >= 1.2 (20% improvement)',
                'agricultural_interpretation': 'Domain-specific categorization and analysis'
            }
        }
        
        # Save detailed report
        output_dir = self.results_dir / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "agricultural_insights_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create executive summary
        executive_summary = self._create_executive_summary(report)
        with open(output_dir / "executive_summary.txt", 'w') as f:
            f.write(executive_summary)
        
        # Create CSV summaries for key insights
        self._save_csv_summaries(output_dir)
        
        print(f"✅ Insights report saved to: {output_dir}")
        print(f"   📄 Full report: agricultural_insights_report.json")
        print(f"   📋 Executive summary: executive_summary.txt")
        print(f"   📊 CSV summaries: various summary files")
        
        return output_dir
    
    # Helper methods for analysis
    def _interpret_yield_pattern(self, conditions):
        """Interpret yield pattern conditions"""
        conditions_list = [cond.strip() for cond in conditions.split(',')]
        
        soil_factors = [c for c in conditions_list if any(x in c.lower() for x in ['soil', 'ph', 'nitrogen', 'phosphorus', 'potassium'])]
        climate_factors = [c for c in conditions_list if any(x in c.lower() for x in ['rainfall', 'temperature', 'humidity'])]
        management_factors = [c for c in conditions_list if any(x in c.lower() for x in ['fertilizer', 'irrigation', 'tillage'])]
        
        interpretation = []
        if soil_factors:
            interpretation.append(f"Optimal soil conditions: {', '.join(soil_factors[:2])}")
        if climate_factors:
            interpretation.append(f"Favorable climate: {', '.join(climate_factors[:2])}")
        if management_factors:
            interpretation.append(f"Best practices: {', '.join(management_factors[:2])}")
        
        return '; '.join(interpretation) if interpretation else "Multiple favorable conditions"
    
    def _extract_profit_strategies(self, profit_rules):
        """Extract top profit strategies"""
        strategies = []
        for idx, rule in profit_rules.head(5).iterrows():
            strategy = {
                'description': self._simplify_rule_description(rule['antecedents']),
                'confidence': rule['confidence'],
                'lift': rule['lift'],
                'business_impact': self._assess_business_impact(rule)
            }
            strategies.append(strategy)
        return strategies
    
    def _extract_cost_factors(self, cost_rules):
        """Extract cost optimization factors"""
        all_factors = []
        for rule in cost_rules['antecedents']:
            factors = [factor.strip() for factor in rule.split(',')]
            all_factors.extend(factors)
        
        factor_frequency = pd.Series(all_factors).value_counts()
        return factor_frequency.head(10).to_dict()
    
    def _interpret_sustainability_benefit(self, rule):
        """Interpret sustainability benefits"""
        if 'water_efficiency' in rule['consequents']:
            return "Improved water conservation and efficiency"
        elif 'carbon_efficiency' in rule['consequents']:
            return "Reduced carbon footprint and greenhouse gas emissions"
        elif 'sustainability_level' in rule['consequents']:
            return "Overall sustainable farming practices adoption"
        else:
            return "General environmental benefit"
    
    def _identify_eco_combinations(self, sustainability_rules):
        """Identify eco-friendly farming combinations"""
        combinations = []
        for idx, rule in sustainability_rules.head(5).iterrows():
            combo = {
                'description': self._simplify_rule_description(rule['antecedents']),
                'confidence': rule['confidence'],
                'lift': rule['lift'],
                'impact': self._interpret_sustainability_benefit(rule)
            }
            combinations.append(combo)
        return combinations
    
    def _extract_crops_from_rule(self, antecedents):
        """Extract crop types from rule antecedents"""
        crops = []
        conditions = [cond.strip() for cond in antecedents.split(',')]
        for condition in conditions:
            if 'crop_type_' in condition:
                crop = condition.replace('crop_type_', '').strip()
                crops.append(crop)
        return crops
    
    def _extract_soil_conditions_from_rule(self, antecedents):
        """Extract soil conditions from rule antecedents"""
        soil_conditions = []
        conditions = [cond.strip() for cond in antecedents.split(',')]
        for condition in conditions:
            if any(x in condition.lower() for x in ['soil_type_', 'ph_', 'organicmatter_', 'nitrogen_', 'phosphorus_', 'potassium_']):
                soil_conditions.append(condition)
        return ', '.join(soil_conditions)
    
    def _find_versatile_soils(self, crop_soil_rules):
        """Find most versatile soil types"""
        soil_mentions = []
        for rule in crop_soil_rules['antecedents']:
            conditions = [cond.strip() for cond in rule.split(',')]
            for condition in conditions:
                if 'soil_type_' in condition:
                    soil_mentions.append(condition)
        
        soil_frequency = pd.Series(soil_mentions).value_counts()
        return soil_frequency.head(5).to_dict()
    
    def _find_adaptable_crops(self, crop_soil_rules):
        """Find most adaptable crops"""
        crop_mentions = []
        for rule in crop_soil_rules['antecedents']:
            conditions = [cond.strip() for cond in rule.split(',')]
            for condition in conditions:
                if 'crop_type_' in condition:
                    crop_mentions.append(condition)
        
        crop_frequency = pd.Series(crop_mentions).value_counts()
        return crop_frequency.head(5).to_dict()
    
    def _extract_regions_from_rules(self, regional_rules):
        """Extract unique regions from rules"""
        regions = set()
        for rule in regional_rules['antecedents']:
            conditions = [cond.strip() for cond in rule.split(',')]
            for condition in conditions:
                if 'region_' in condition:
                    region = condition.replace('region_', '').strip()
                    regions.add(region)
        return list(regions)
    
    def _extract_regional_practices(self, region_rules, region):
        """Extract characteristic practices for a region"""
        practices = []
        for idx, rule in region_rules.head(3).iterrows():
            practice = self._simplify_rule_description(rule['antecedents'])
            practices.append(practice)
        return practices
    
    def _extract_regional_performance(self, region_rules):
        """Extract performance indicators for region"""
        performance_outcomes = []
        for rule in region_rules['consequents']:
            if any(x in rule.lower() for x in ['yield_high', 'profit_high', 'efficiency_high']):
                performance_outcomes.append(rule)
        
        return list(set(performance_outcomes))[:3]
    
    def _extract_climate_adaptations(self, region_rules):
        """Extract climate adaptation strategies"""
        adaptations = []
        for rule in region_rules['antecedents']:
            conditions = [cond.strip() for cond in rule.split(',')]
            climate_conditions = [c for c in conditions if any(x in c.lower() for x in ['rainfall', 'temperature', 'irrigation'])]
            if climate_conditions:
                adaptations.extend(climate_conditions)
        
        return list(set(adaptations))[:3]
    
    def _generate_yield_recommendations(self):
        """Generate yield optimization recommendations"""
        recommendations = []
        if 'high_yield_patterns' in self.agricultural_insights:
            top_factors = self.agricultural_insights['high_yield_patterns'].get('most_important_factors', {})
            for factor, frequency in list(top_factors.items())[:5]:
                if 'soil_type_' in factor:
                    recommendations.append(f"Optimize soil type selection: {factor.replace('soil_type_', '')}")
                elif 'fertilizer_type_' in factor:
                    recommendations.append(f"Use appropriate fertilization: {factor.replace('fertilizer_type_', '')}")
                elif 'irrigation_method_' in factor:
                    recommendations.append(f"Implement efficient irrigation: {factor.replace('irrigation_method_', '')}")
        
        return recommendations[:5]
    
    def _generate_profit_recommendations(self):
        """Generate profit maximization recommendations"""
        recommendations = []
        if 'profitability_patterns' in self.agricultural_insights:
            strategies = self.agricultural_insights['profitability_patterns'].get('top_profit_strategies', [])
            for strategy in strategies[:3]:
                recommendations.append(f"Implement: {strategy['description']}")
        
        return recommendations
    
    def _generate_sustainability_recommendations(self):
        """Generate sustainability improvement recommendations"""
        recommendations = []
        if 'sustainability_patterns' in self.agricultural_insights:
            practices = self.agricultural_insights['sustainability_patterns'].get('eco_friendly_combinations', [])
            for practice in practices[:3]:
                recommendations.append(f"Adopt sustainable practice: {practice['description']}")
        
        return recommendations
    
    def _generate_crop_selection_guide(self):
        """Generate crop selection guide"""
        guide = []
        if 'crop_soil_combinations' in self.agricultural_insights:
            recommendations = self.agricultural_insights['crop_soil_combinations'].get('crop_specific_recommendations', {})
            for crop, recs in list(recommendations.items())[:3]:
                if recs:
                    guide.append(f"{crop}: Best with {recs[0]['soil_conditions']}")
        
        return guide
    
    def _generate_regional_recommendations(self):
        """Generate regional best practices"""
        recommendations = []
        if 'regional_patterns' in self.agricultural_insights:
            regional_data = self.agricultural_insights['regional_patterns']
            for region, data in list(regional_data.items())[:3]:
                if data.get('characteristic_practices'):
                    recommendations.append(f"{region}: {data['characteristic_practices'][0]}")
        
        return recommendations
    
    def _simplify_rule_description(self, antecedents):
        """Simplify rule description for readability"""
        conditions = [cond.strip() for cond in antecedents.split(',')]
        simplified = []
        
        for condition in conditions[:3]:  # Show top 3 conditions
            if '_' in condition:
                parts = condition.split('_')
                if len(parts) >= 2:
                    simplified.append(f"{parts[0].replace('_', ' ')}: {parts[-1]}")
                else:
                    simplified.append(condition)
            else:
                simplified.append(condition)
        
        return ', '.join(simplified)
    
    def _assess_business_impact(self, rule):
        """Assess business impact of a rule"""
        if rule['lift'] > 3.0:
            return "High Impact"
        elif rule['lift'] > 2.0:
            return "Medium Impact"
        else:
            return "Low Impact"
    
    def _prepare_dashboard_data(self):
        """Prepare data for dashboard visualization"""
        # This would prepare the data structures needed for the dashboard
        # For brevity, returning a placeholder structure
        return {
            'yield_factors': self.agricultural_insights.get('high_yield_patterns', {}).get('most_important_factors', {}),
            'profitability_data': self.agricultural_insights.get('profitability_patterns', {}),
            'sustainability_metrics': self.agricultural_insights.get('sustainability_patterns', {}),
            'crop_soil_data': self.agricultural_insights.get('crop_soil_combinations', {}),
            'regional_performance': self.agricultural_insights.get('regional_patterns', {}),
            'rule_categories': self.rule_analysis.get('category_distribution', {}) if self.rule_analysis else {}
        }
    
    def _create_matplotlib_dashboard(self):
        """Create dashboard using matplotlib"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Agricultural Association Rules - Insights Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Yield Success Factors
        if 'high_yield_patterns' in self.agricultural_insights:
            factors = self.agricultural_insights['high_yield_patterns'].get('most_important_factors', {})
            if factors:
                top_factors = dict(list(factors.items())[:5])
                axes[0, 0].bar(range(len(top_factors)), list(top_factors.values()), color='green', alpha=0.7)
                axes[0, 0].set_xticks(range(len(top_factors)))
                axes[0, 0].set_xticklabels([k.split('_')[-1] for k in top_factors.keys()], rotation=45)
                axes[0, 0].set_title('Top Yield Success Factors')
                axes[0, 0].set_ylabel('Frequency in Rules')
        
        # 2. Rule Categories Distribution
        if self.rule_analysis and 'category_distribution' in self.rule_analysis:
            categories = self.rule_analysis['category_distribution']
            axes[0, 1].pie(categories.values(), labels=categories.keys(), autopct='%1.1f%%')
            axes[0, 1].set_title('Rule Categories Distribution')
        
        # 3. Profitability Metrics
        if 'profitability_patterns' in self.agricultural_insights:
            profit_data = self.agricultural_insights['profitability_patterns']
            metrics = ['total_profit_rules', 'cost_efficiency_rules']
            values = [profit_data.get(m, 0) for m in metrics]
            axes[0, 2].bar(metrics, values, color='gold', alpha=0.7)
            axes[0, 2].set_title('Profitability Rule Counts')
            axes[0, 2].set_ylabel('Number of Rules')
        
        # 4. Sustainability Patterns
        if 'sustainability_patterns' in self.agricultural_insights:
            sustain_data = self.agricultural_insights['sustainability_patterns']
            axes[1, 0].bar(['Total Rules'], [sustain_data.get('total_sustainability_rules', 0)], 
                          color='lightgreen', alpha=0.7)
            axes[1, 0].set_title('Sustainability Rules')
            axes[1, 0].set_ylabel('Number of Rules')
        
        # 5. Crop-Soil Combinations
        if 'crop_soil_combinations' in self.agricultural_insights:
            combo_data = self.agricultural_insights['crop_soil_combinations']
            versatile_soils = combo_data.get('most_versatile_soils', {})
            if versatile_soils:
                top_soils = dict(list(versatile_soils.items())[:5])
                axes[1, 1].bar(range(len(top_soils)), list(top_soils.values()), color='brown', alpha=0.7)
                axes[1, 1].set_xticks(range(len(top_soils)))
                axes[1, 1].set_xticklabels([k.split('_')[-1] for k in top_soils.keys()], rotation=45)
                axes[1, 1].set_title('Most Versatile Soils')
                axes[1, 1].set_ylabel('Frequency')
        
        # 6. Regional Patterns
        if 'regional_patterns' in self.agricultural_insights:
            regional_data = self.agricultural_insights['regional_patterns']
            region_counts = {region: data.get('total_rules', 0) for region, data in regional_data.items()}
            if region_counts:
                axes[1, 2].bar(range(len(region_counts)), list(region_counts.values()), color='skyblue', alpha=0.7)
                axes[1, 2].set_xticks(range(len(region_counts)))
                axes[1, 2].set_xticklabels([r.replace('_', ' ') for r in region_counts.keys()], rotation=45)
                axes[1, 2].set_title('Regional Rule Distribution')
                axes[1, 2].set_ylabel('Number of Rules')
        
        plt.tight_layout()
        
        # Save dashboard
        output_dir = self.results_dir / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "agricultural_insights_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Dashboard saved: agricultural_insights_dashboard.png")
        return fig
    
    # Placeholder methods for Plotly dashboard components
    def _add_yield_factors_plot(self, fig, data, row, col):
        """Add yield factors plot to dashboard"""
        pass
    
    def _add_profitability_plot(self, fig, data, row, col):
        """Add profitability plot to dashboard"""
        pass
    
    def _add_sustainability_plot(self, fig, data, row, col):
        """Add sustainability plot to dashboard"""
        pass
    
    def _add_crop_soil_heatmap(self, fig, data, row, col):
        """Add crop-soil heatmap to dashboard"""
        pass
    
    def _add_regional_performance(self, fig, data, row, col):
        """Add regional performance plot to dashboard"""
        pass
    
    def _add_rule_categories_pie(self, fig, data, row, col):
        """Add rule categories pie chart to dashboard"""
        pass
    
    def _create_executive_summary(self, report):
        """Create executive summary text"""
        summary = f"""
AGRICULTURAL ASSOCIATION RULES ANALYSIS - EXECUTIVE SUMMARY
===========================================================

Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

OVERVIEW
--------
This analysis discovered {report['summary']['total_rules_analyzed']} association rules from agricultural data,
revealing key patterns that can guide farming decisions for improved yields, profitability, and sustainability.

KEY FINDINGS
------------
• High-Yield Patterns: {report['summary']['high_yield_rules']} rules identified optimal conditions for maximum crop yields
• Profitability Rules: {report['summary']['profitability_rules']} rules show pathways to improved farm profitability  
• Sustainability Practices: {report['summary']['sustainability_rules']} rules demonstrate eco-friendly farming methods
• Crop-Soil Combinations: {report['summary']['crop_soil_combinations']} optimal crop-soil pairings discovered

TOP RECOMMENDATIONS
-------------------
"""
        
        # Add top recommendations
        if 'recommendations' in report['key_insights']:
            recs = report['key_insights']['recommendations']
            
            if 'yield_optimization' in recs:
                summary += "\nYield Optimization:\n"
                for rec in recs['yield_optimization'][:3]:
                    summary += f"• {rec}\n"
            
            if 'profit_maximization' in recs:
                summary += "\nProfit Maximization:\n"
                for rec in recs['profit_maximization'][:3]:
                    summary += f"• {rec}\n"
            
            if 'sustainability_improvement' in recs:
                summary += "\nSustainability Improvement:\n"
                for rec in recs['sustainability_improvement'][:3]:
                    summary += f"• {rec}\n"
        
        summary += f"""

METHODOLOGY
-----------
• Data Mining Technique: Association Rules Mining using Apriori Algorithm
• Confidence Threshold: 60% minimum confidence for reliable patterns
• Lift Threshold: 1.2+ for meaningful associations (20% improvement over random)
• Agricultural Focus: Domain-specific interpretation and categorization

BUSINESS IMPACT
---------------
These insights enable farmers to:
1. Make data-driven decisions for crop selection and management
2. Optimize resource allocation for maximum ROI
3. Implement sustainable practices without sacrificing productivity
4. Adapt farming strategies to regional conditions

For detailed analysis and specific recommendations, see the full report.
"""
        return summary
    
    def _save_csv_summaries(self, output_dir):
        """Save CSV summaries of key insights"""
        
        # High-yield patterns summary
        if 'high_yield_patterns' in self.agricultural_insights:
            yield_data = self.agricultural_insights['high_yield_patterns']
            if 'top_patterns' in yield_data:
                yield_df = pd.DataFrame(yield_data['top_patterns'])
                yield_df.to_csv(output_dir / "high_yield_patterns.csv", index=False)
        
        # Profitability strategies summary
        if 'profitability_patterns' in self.agricultural_insights:
            profit_data = self.agricultural_insights['profitability_patterns']
            if 'top_profit_strategies' in profit_data:
                profit_df = pd.DataFrame(profit_data['top_profit_strategies'])
                profit_df.to_csv(output_dir / "profitability_strategies.csv", index=False)
        
        # Sustainability practices summary
        if 'sustainability_patterns' in self.agricultural_insights:
            sustain_data = self.agricultural_insights['sustainability_patterns']
            if 'sustainable_practices' in sustain_data:
                sustain_df = pd.DataFrame(sustain_data['sustainable_practices'])
                sustain_df.to_csv(output_dir / "sustainability_practices.csv", index=False)
        
        # Crop recommendations summary
        if 'crop_soil_combinations' in self.agricultural_insights:
            crop_data = self.agricultural_insights['crop_soil_combinations']
            if 'crop_specific_recommendations' in crop_data:
                crop_recs = []
                for crop, recommendations in crop_data['crop_specific_recommendations'].items():
                    for rec in recommendations:
                        crop_recs.append({
                            'crop': crop,
                            'soil_conditions': rec['soil_conditions'],
                            'confidence': rec['confidence'],
                            'lift': rec['lift']
                        })
                if crop_recs:
                    crop_df = pd.DataFrame(crop_recs)
                    crop_df.to_csv(output_dir / "crop_soil_recommendations.csv", index=False)

def main():
    """
    Main analysis function for Step 5
    """
    print("📊 Agricultural Insights Analysis - Step 5")
    print("=" * 60)
    print("Analyzing association rules and generating agricultural insights")
    print("=" * 60)
    
    try:
        # Initialize analyzer
        analyzer = AgriculturalInsightsAnalyzer()
        
        # Load mining results
        if not analyzer.load_mining_results():
            print("❌ Could not load mining results. Make sure Step 4 completed successfully.")
            return None
        
        print(f"\n🔍 Performing Agricultural Analysis...")
        
        # Perform all analyses
        yield_insights = analyzer.analyze_high_yield_patterns()
        profit_insights = analyzer.analyze_profitability_patterns()
        sustainability_insights = analyzer.analyze_sustainability_patterns()
        crop_soil_insights = analyzer.analyze_crop_soil_combinations()
        regional_insights = analyzer.analyze_regional_patterns()
        
        # Generate recommendations
        recommendations = analyzer.generate_recommendations()
        
        # Create dashboard
        dashboard = analyzer.create_insights_dashboard()
        
        # Save comprehensive report
        report_dir = analyzer.save_insights_report()
        
        print(f"\n🎉 Step 5 Complete!")
        print(f"📁 Insights report saved to: {report_dir}")
        print(f"📊 Dashboard created with visualizations")
        print(f"📋 Actionable recommendations generated")
        print("✅ Agricultural insights analysis completed successfully!")
        print("=" * 60)
        
        # Display final summary
        print(f"\n📈 FINAL ANALYSIS SUMMARY:")
        print(f"   🌾 High-yield patterns analyzed")
        print(f"   💰 Profitability strategies identified")
        print(f"   🌱 Sustainability practices discovered")
        print(f"   🌍 Regional patterns analyzed")
        print(f"   📋 Actionable recommendations generated")
        
        return analyzer
        
    except Exception as e:
        print(f"❌ Insights analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    # Run agricultural insights analysis
    analyzer = main()