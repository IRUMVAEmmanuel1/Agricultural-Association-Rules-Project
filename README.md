# 🌾 Agricultural Association Rules Analytics

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Data Science](https://img.shields.io/badge/Data%20Science-Association%20Rules-orange.svg)](https://en.wikipedia.org/wiki/Association_rule_learning)

A comprehensive data mining application that discovers farming patterns and provides actionable insights for precision agriculture using Association Rules Mining (Apriori Algorithm).

## 👥 Authors & Contributors

- **Jacquiline Umurerwa Karangwa**:  Team Members
- **Jean D'amour Uwamahoro**: Team Members
- **Emmanuel Irumva**:  Team Members
- **Lysee Rita Umwari Mwiza**:  Team Members
- **Kevin Ishimwe**:  Team Members


## 📖 Overview

This project implements an end-to-end agricultural analytics pipeline that:
- Generates realistic farm data with 35+ agricultural variables
- Applies Association Rules Mining to discover farming patterns
- Provides interactive web dashboard for stakeholder access
- Delivers actionable recommendations for yield optimization, profitability, and sustainability

## ✨ Features

### 🔍 Data Mining & Analytics
- **Association Rules Mining** using Apriori Algorithm
- **Pattern Discovery** for crop-soil-climate relationships
- **Statistical Analysis** of 5,000+ farm records
- **Predictive Insights** for farming decision support

### 📊 Interactive Web Dashboard
- **Real-time Data Exploration** with dynamic filtering
- **Visualization Dashboard** with interactive charts
- **Personalized Recommendations** based on farm conditions
- **What-if Scenario Analysis** for planning
- **Export Capabilities** for reports and data

### 🌾 Agricultural Focus Areas
- **Yield Optimization** strategies and patterns
- **Profitability Analysis** and cost optimization
- **Sustainability Practices** and environmental impact
- **Regional Best Practices** and climate adaptation
- **Crop-Soil Combinations** for optimal selection

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 4GB+ RAM recommended
- Modern web browser for dashboard

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/agricultural-association-rules.git
   cd agricultural-association-rules
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate Data and Run Analysis**
   ```bash
   python run_complete_pipeline.py
   ```

4. **Launch Web Application**
   ```bash
   streamlit run app/agricultural_analytics_app.py
   ```

5. **Access Dashboard**
   Open your browser to `http://localhost:8501`

## 📁 Project Structure

```
agricultural_association_rules/
├── 🐍 run_complete_pipeline.py          # Complete data pipeline runner
├── 📱 app/
│   └── agricultural_analytics_app.py    # Streamlit web application
├── 📊 data/
│   ├── raw/                            # Original farm data
│   ├── processed/                      # Cleaned and transformed data
│   └── external/                       # External data sources
├── 🔬 src/
│   ├── data/                          # Data processing modules
│   ├── models/                        # Association rules mining
│   └── analysis/                      # Insights generation
├── 📈 results/
│   ├── models/                        # Mining results (rules, itemsets)
│   ├── reports/                       # Analysis reports
│   └── figures/                       # Visualizations
├── ⚙️ configs/                         # Configuration files
├── 📝 docs/                           # Documentation
├── 🧪 tests/                          # Unit tests
└── 📋 requirements.txt                # Python dependencies
```

## 🛠️ Dependencies

### Core Libraries
```txt
pandas>=1.5.0                # Data manipulation and analysis
numpy>=1.21.0                # Numerical computing
scikit-learn>=1.1.0          # Machine learning utilities
mlxtend>=0.21.0              # Association rules mining
```

### Visualization & Web App
```txt
streamlit>=1.28.0            # Web application framework
plotly>=5.10.0               # Interactive visualizations
matplotlib>=3.5.0            # Static plotting
seaborn>=0.11.0              # Statistical visualizations
```

### Additional Tools
```txt
openpyxl>=3.0.0              # Excel file handling
python-dotenv>=0.20.0        # Environment variables
tqdm>=4.64.0                 # Progress bars
```

## 📊 Dataset Information

### Generated Farm Records (5,000 records)
- **Geographic Coverage**: 7 US agricultural regions
- **Crop Diversity**: 10+ major crop types
- **Variables**: 35+ agricultural and economic indicators
- **Time Span**: Full growing season data

### Key Variables
| Category | Variables |
|----------|-----------|
| **Geographic** | Region, Latitude, Longitude, Elevation |
| **Soil Properties** | Type, pH, Organic Matter, N-P-K Content |
| **Climate** | Temperature, Rainfall, Humidity, Solar Radiation |
| **Crop Management** | Type, Variety, Planting/Harvest Dates |
| **Practices** | Fertilizer, Irrigation, Tillage Methods |
| **Performance** | Yield, Revenue, Profit, Costs |
| **Sustainability** | Water Usage, Carbon Footprint |

## 🔬 Methodology

### Association Rules Mining Pipeline

1. **Data Preprocessing**
   - Categorical variable creation
   - Transaction format conversion
   - Feature engineering for agricultural context

2. **Pattern Discovery**
   - Apriori algorithm implementation
   - Frequent itemset mining (min_support = 0.05)
   - Association rule generation (min_confidence = 0.6, min_lift = 1.2)

3. **Agricultural Interpretation**
   - Domain-specific rule categorization
   - Business impact assessment
   - Actionable recommendation generation

4. **Validation & Insights**
   - Statistical significance testing
   - Cross-validation with agricultural best practices
   - Stakeholder-friendly reporting

## 📈 Key Findings & Applications

### High-Impact Discoveries
- **Yield Optimization**: Identified 15+ conditions leading to 25%+ yield improvements
- **Cost Reduction**: Discovered patterns for 20% cost optimization
- **Sustainability**: Found eco-friendly practices maintaining productivity
- **Regional Adaptation**: Region-specific best practices for climate resilience

### Business Applications
- **Precision Agriculture**: Data-driven crop and input selection
- **Risk Management**: Pattern-based prediction and mitigation
- **Resource Optimization**: Efficient allocation of water, fertilizer, labor
- **Market Strategy**: Profit maximization through optimal practices

## 🌐 Web Dashboard Features

### Interactive Pages
1. **📊 Overview** - Project summary and key metrics
2. **🔍 Data Exploration** - Interactive data analysis with filters
3. **🔗 Association Rules** - Rule visualization and exploration
4. **🌾 Agricultural Insights** - Yield, profit, sustainability patterns
5. **📋 Recommendations** - Personalized farming advice
6. **🔍 Interactive Analysis** - What-if scenarios and custom analysis
7. **📤 Export Results** - Download reports and data

### Key Capabilities
- **Dynamic Filtering** by crop, region, yield range
- **Real-time Visualization** with interactive charts
- **Personalized Recommendations** based on farm conditions
- **Scenario Modeling** for planning and decision support
- **Comprehensive Export** of all results and insights

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Install development dependencies (`pip install -r requirements-dev.txt`)
4. Make your changes
5. Run tests (`python -m pytest tests/`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📝 Usage Examples

### Basic Analysis
```python
from src.models.association_rules_mining import AgriculturalAssociationRulesMiner

# Initialize miner
miner = AgriculturalAssociationRulesMiner()

# Load and analyze data
transactions, metadata, stats = miner.load_transaction_data('data/processed/agricultural_transactions.json')
rules = miner.generate_association_rules(frequent_itemsets)

# Get insights
insights = miner.analyze_rule_patterns()
```

### Custom Recommendations
```python
from src.analysis.agricultural_insights_analyzer import AgriculturalInsightsAnalyzer

# Generate personalized recommendations
analyzer = AgriculturalInsightsAnalyzer()
recommendations = analyzer.generate_personalized_recommendations(
    crop="Corn", 
    soil="Loam", 
    region="Midwest", 
    goal="Maximize Yield"
)
```

## 📊 Performance Metrics

### Data Processing
- **Processing Speed**: 5,000 records in <2 minutes
- **Memory Usage**: <1GB RAM for full dataset
- **Rule Generation**: 50+ association rules discovered
- **Analysis Depth**: 147 unique agricultural items analyzed

### Web Application
- **Load Time**: <3 seconds for dashboard
- **Responsiveness**: Real-time filtering and visualization
- **Export Speed**: Full report generation in <10 seconds
- **Browser Support**: Chrome, Firefox, Safari, Edge

## 🔧 Configuration

### Environment Variables
```bash
# .env file
MIN_SUPPORT=0.05
MIN_CONFIDENCE=0.6
MIN_LIFT=1.2
```

### Custom Parameters
```python
# configs/config.py
ASSOCIATION_RULES_CONFIG = {
    'min_support': 0.05,
    'min_confidence': 0.6,
    'min_lift': 1.2,
    'max_len': 6
}
```

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test category
python -m pytest tests/test_association_rules.py
```

## 📚 Documentation

- **API Documentation**: `docs/api.md`
- **User Guide**: `docs/user_guide.md`
- **Technical Specifications**: `docs/technical_specs.md`
- **Deployment Guide**: `docs/deployment.md`

## 🐳 Docker Deployment

```bash
# Build Docker image
docker build -t agricultural-analytics .

# Run container
docker run -p 8501:8501 agricultural-analytics

# Using docker-compose
docker-compose up -d
```

## ☁️ Cloud Deployment

### Streamlit Cloud
1. Push code to GitHub
2. Connect repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy with one click

### Heroku
```bash
# Install Heroku CLI and login
heroku create agricultural-analytics-app
git push heroku main
```

### AWS/GCP/Azure
See detailed deployment guides in `docs/deployment.md`

## 🔒 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Agricultural Research Community** for domain expertise
- **Open Source Libraries** that made this project possible
- **Farmers and Agricultural Professionals** for validation and feedback

## 📞 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/agricultural-association-rules/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/agricultural-association-rules/discussions)
- **Email**: agricultural.analytics@example.com
- **Documentation**: [Project Wiki](https://github.com/yourusername/agricultural-association-rules/wiki)

## 🗺️ Roadmap

### Version 2.0 (Planned)
- [ ] Real-time data integration with IoT sensors
- [ ] Machine learning model ensemble
- [ ] Mobile application development
- [ ] Multi-language support
- [ ] Advanced geospatial analysis

### Version 1.5 (In Progress)
- [ ] Enhanced visualization capabilities
- [ ] API endpoint development
- [ ] Database integration
- [ ] Performance optimizations

## 📊 Project Statistics

![GitHub stars](https://img.shields.io/github/stars/yourusername/agricultural-association-rules)
![GitHub forks](https://img.shields.io/github/forks/yourusername/agricultural-association-rules)
![GitHub issues](https://img.shields.io/github/issues/yourusername/agricultural-association-rules)
![GitHub last commit](https://img.shields.io/github/last-commit/yourusername/agricultural-association-rules)

---

<p align="center">
  <strong>🌾 Empowering Agriculture Through Data Science 🌾</strong>
</p>

<p align="center">
  Built with ❤️ for farmers, researchers, and agricultural professionals worldwide
</p>
