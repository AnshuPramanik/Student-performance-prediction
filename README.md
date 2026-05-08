# Student Performance Prediction System
contributers - Rohit_Narang, Ravi Shankar
<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Machine Learning](https://img.shields.io/badge/ML-scikit--learn-orange?style=flat-square)
![Framework](https://img.shields.io/badge/Framework-Flask-green?style=flat-square&logo=flask)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

**An intelligent machine learning system to predict student academic performance based on multiple behavioral and contextual factors.**

[Features](#features) • [Quick Start](#quick-start) • [Installation](#installation) • [Usage](#usage) • [Project Structure](#project-structure)

</div>

---

## 📋 Overview

The **Student Performance Prediction System** is a comprehensive machine learning project designed to predict student grades and academic performance. The system integrates data cleaning, exploratory data analysis (EDA), model training, and a user-friendly Flask web application for real-time predictions.

### Key Capabilities:
- 🤖 **ML Model**: Trained predictive models for accurate grade prediction
- 📊 **Data Analysis**: Complete data cleaning and exploratory analysis pipeline
- 🌐 **Web Interface**: Interactive Flask-based web application for predictions
- 📈 **Visualizations**: Comprehensive plots and correlation analysis
- 🔧 **Production Ready**: Serialized models and scalers for deployment

---

## ✨ Features

### Machine Learning
- **Multiple Feature Support**: Analyzes 15+ student attributes
- **Advanced Preprocessing**: Automatic scaling and label encoding
- **Feature Selection**: Optimized feature set for best performance
- **Model Persistence**: Trained models saved for production use

### Supported Features
- Study hours and study methods
- Attendance percentage
- Age and gender demographics
- Subject scores (DAA, Computer Networks, Operating Systems)
- Internet access and travel time
- Parent education level
- Extra-curricular activities
- School type classification

### Web Application
- **Real-time Predictions**: Get performance predictions instantly
- **Beautiful UI**: Modern, responsive dark-themed interface
- **Error Handling**: Comprehensive input validation
- **Fast Response**: Optimized model inference

### Data Analytics
- **EDA Notebooks**: Detailed exploratory analysis
- **Visualizations**: 15+ publication-quality charts
- **Correlation Analysis**: Feature relationship heatmaps
- **Distribution Analysis**: Statistical insights into data

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### 30-Second Setup
```bash
# Clone the repository
git clone https://github.com/AnshuPramanik/student-performance-prediction.git
cd student-performance-prediction

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py

# Open browser to http://localhost:5000
```

---

## 📦 Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/AnshuPramanik/student-performance-prediction.git
cd student-performance-prediction
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Dependencies
- **pandas** (v1.3+) - Data manipulation and analysis
- **numpy** (v1.20+) - Numerical computing
- **scikit-learn** (v0.24+) - Machine learning algorithms
- **matplotlib** (v3.4+) - Data visualization
- **seaborn** (v0.11+) - Statistical visualization
- **flask** (v2.0+) - Web framework

---

## 💻 Usage

### Web Application

#### Starting the Server
```bash
python app.py
```

The application will start at `http://localhost:5000`

#### Making Predictions
1. Open the web interface
2. Fill in the student information form:
   - Study hours (0-10)
   - Study method (Online/Offline/Both)
   - Attendance percentage (0-100)
   - Age (15-25 years)
   - Gender
   - Subject scores
   - And more...
3. Click "Predict Performance"
4. View the predicted grade and confidence score

### Jupyter Notebooks

#### Exploratory Data Analysis
```bash
jupyter notebook eda.ipynb
```
Explore data distributions, correlations, and patterns.

#### Data Cleaning
```bash
jupyter notebook data_cleaning.ipynb
```
Review data preprocessing and cleaning steps.

#### Model Training
```bash
jupyter notebook student_performance_ml.ipynb
```
Train models, evaluate performance, and generate predictions.

---

## 📁 Project Structure

```
student-performance-prediction/
│
├── 📄 README.md                          # Project documentation
├── 📄 requirements.txt                   # Python dependencies
├── 📄 app.py                             # Flask web application
│
├── 📊 Data Files
│   ├── Student_Performance.csv           # Original dataset
│   ├── cleaned_dataset.csv               # Cleaned data
│   └── processed_dataset.csv             # Processed features
│
├── 📓 Notebooks
│   ├── data_cleaning.ipynb               # Data preprocessing pipeline
│   ├── eda.ipynb                         # Exploratory data analysis
│   └── student_performance_ml.ipynb      # Model training & evaluation
│
├── 🤖 Trained Models (Pickled)
│   ├── best_model.pkl                    # Trained ML model
│   ├── scaler.pkl                        # Feature scaler
│   ├── label_encoder.pkl                 # Categorical encoder
│   ├── selected_features.pkl             # Feature names
│   └── numerical_selected.pkl            # Numerical features list
│
├── 📈 Visualizations
│   ├── plots/                            # Generated plots directory
│   ├── correlation_heatmap.png           # Feature correlations
│   ├── *_distribution.png                # Distribution plots
│   ├── *_vs_grade.png                    # Feature vs grade relationships
│   └── Confusion Metrics.png             # Model performance
│
└── 🛠️ Utilities
    ├── add_accuracy_cell.py              # Accuracy calculation helper
    └── .git/                             # Version control
```

---

## 🔍 Data Description

### Dataset
- **Size**: Multiple student records with 15+ features
- **Target Variable**: Student Grade/Performance
- **Features**: Mix of categorical and numerical attributes

### Key Features
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| Study Hours | Numerical | 0-10 | Weekly study hours |
| Attendance | Numerical | 0-100 | Attendance percentage |
| DAA Score | Numerical | 0-100 | Design and Analysis of Algorithms |
| Computer Networks | Numerical | 0-100 | Computer Networks score |
| OS Score | Numerical | 0-100 | Operating Systems score |
| Age | Numerical | 15-25 | Student age |
| Study Method | Categorical | 3 types | Online/Offline/Both |
| Gender | Categorical | 2 types | Male/Female |
| Parent Education | Categorical | Multiple | Parent's education level |
| Internet Access | Categorical | Yes/No | Internet availability |
| School Type | Categorical | Multiple | Type of school |

---

## 🧠 Model Information

### Algorithm Details
- **Primary Model**: Gradient Boosting / Random Forest
- **Preprocessing**: StandardScaler for numerical features
- **Encoding**: LabelEncoder for categorical features
- **Feature Selection**: Optimized feature subset for performance

### Model Performance
- **Accuracy**: High prediction accuracy on validation set
- **Cross-validation**: K-fold validation employed
- **Metrics**: Precision, Recall, F1-Score evaluated

### Inference
```python
# Example prediction code
import pickle
import pandas as pd

# Load model and preprocessors
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare input data
input_data = pd.DataFrame({...})
scaled_data = scaler.transform(input_data)

# Make prediction
prediction = model.predict(scaled_data)
```

---

## 📊 Key Visualizations

The project includes comprehensive visualizations:

### Exploratory Analysis
- **Correlation Heatmap**: Feature relationships and multicollinearity
- **Distribution Plots**: Individual feature distributions
- **Count Plots**: Categorical feature distributions
- **Pair Plots**: Multi-feature relationships

### Performance Analysis
- **Feature vs Grade**: Scatter plots showing feature-grade relationships
- **Confusion Matrix**: Model classification metrics
- **Feature Importance**: Top contributing features visualization

---

## 🔧 Configuration

### Model Settings
Models and preprocessors are automatically loaded from pickle files in the root directory. No configuration file needed for basic usage.

### Flask Settings
```python
# Default Flask configuration (in app.py)
app = Flask(__name__)
app.run(debug=True, port=5000)
```

---

## 🚀 Deployment

### Local Deployment
```bash
python app.py
```

### Production Deployment (Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker Deployment
```bash
docker build -t student-performance .
docker run -p 5000:5000 student-performance
```

---

## 📈 Workflow

### Step-by-Step Process

#### 1. **Data Collection** 
   - Source: Original dataset (Student_Performance.csv)
   - Records: Multiple student entries
   - Features: 15+ attributes

#### 2. **Data Cleaning** (data_cleaning.ipynb)
   - Handle missing values
   - Remove outliers
   - Data type conversion
   - Output: cleaned_dataset.csv

#### 3. **Exploratory Data Analysis** (eda.ipynb)
   - Statistical summaries
   - Distribution analysis
   - Correlation analysis
   - Visualization generation
   - Identify patterns and relationships

#### 4. **Feature Engineering** (student_performance_ml.ipynb)
   - Feature selection
   - Categorical encoding
   - Numerical scaling
   - Output: processed_dataset.csv

#### 5. **Model Training**
   - Train multiple algorithms
   - Hyperparameter tuning
   - Cross-validation
   - Model evaluation
   - Select best performer

#### 6. **Model Serialization**
   - Save trained model: best_model.pkl
   - Save scaler: scaler.pkl
   - Save encoder: label_encoder.pkl
   - Save feature lists: selected_features.pkl

#### 7. **Web Application Deployment**
   - Load serialized models
   - Set up Flask routes
   - Create user interface
   - Enable real-time predictions

---

## 🛠️ Advanced Usage

### Retraining the Model
```bash
# Edit and run the notebook
jupyter notebook student_performance_ml.ipynb

# After training, models are automatically serialized
# Replace pickle files in root directory
```

### Adding New Features
1. Update data collection pipeline
2. Add to data_cleaning.ipynb
3. Include in feature engineering
4. Retrain model
5. Update form fields in app.py

### Model Evaluation
```python
# In notebook, after training:
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
```

---

## 🔐 Important Notes

- **Model Files**: Required for predictions. Don't commit data-heavy files to git.
- **Virtual Environment**: Always use venv to avoid dependency conflicts
- **Data Privacy**: Ensure compliance with data protection regulations
- **Model Updates**: Retrain periodically with new data for accuracy maintenance

---

## 📝 Data Files Summary

| File | Purpose | Size |
|------|---------|------|
| Student_Performance.csv | Raw dataset | Original |
| cleaned_dataset.csv | After preprocessing | Cleaned |
| processed_dataset.csv | Feature-engineered | Ready for ML |
| best_model.pkl | Trained model | ~500KB |
| scaler.pkl | Numerical scaler | ~10KB |
| label_encoder.pkl | Categorical encoder | ~5KB |

---

## 🤝 Contributing

We welcome contributions! Here's how to help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Contributing Guidelines
- Follow PEP 8 style guide
- Add comments for complex logic
- Update README for new features
- Test thoroughly before submitting PR

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: "No module named 'flask'"
```bash
Solution: pip install flask
```

**Issue**: "Model file not found"
```bash
Solution: Ensure pickle files are in root directory
```

**Issue**: Port 5000 already in use
```bash
Solution: python app.py --port 5001
```

**Issue**: Module import errors
```bash
Solution: Verify virtual environment is activated
Ensure all dependencies installed: pip install -r requirements.txt
```

---

## 📚 Learning Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Machine Learning Guide](https://developers.google.com/machine-learning)

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Anshu Pramanik**
- GitHub: [@AnshuPramanik](https://github.com/AnshuPramanik)
- Email: pramanikanshu456@gmail.com
- LinkedIn: [Anshu Pramanik](https://www.linkedin.com/in/anshu-pramanik-287591267/)

---

## 🙏 Acknowledgments

- Dataset contributors and providers
- Open-source community
- Flask and scikit-learn maintainers
- Contributors and reviewers

---

## 📞 Support

For questions or issues, please:
- 📧 Open an [Issue](https://github.com/AnshuPramanik/student-performance-prediction/issues)
- 💬 Start a [Discussion](https://github.com/AnshuPramanik/student-performance-prediction/discussions)
- 📝 Check [Existing Issues](https://github.com/AnshuPramanik/student-performance-prediction/issues)

---

## 📊 Project Statistics

- **Total Notebooks**: 3
- **Data Files**: 3 CSV files
- **Visualizations**: 15+ charts
- **Model Files**: 5 pickle files
- **Dependencies**: 6 major packages
- **Code Lines**: 1000+

---

<div align="center">

**Made with ❤️ for the ML Community**

⭐ If you find this project helpful, please consider giving it a star!

</div>
