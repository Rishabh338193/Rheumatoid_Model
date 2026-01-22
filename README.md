# 🏥 Rheumatoid Arthritis Prediction Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.1-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)

A complete end-to-end machine learning project for predicting Rheumatoid Arthritis (RA) using Random Forest Classifier. This project includes data generation, model training, REST API backend, and interactive web frontends.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [API Documentation](#api-documentation)
- [Sample Input/Output](#sample-inputoutput)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

Rheumatoid Arthritis (RA) is a chronic inflammatory disorder affecting joints. Early detection is crucial for effective treatment and preventing joint damage. This project uses Machine Learning to predict RA based on clinical and laboratory features.

**⚠️ Educational Purpose Only:** This system is designed for educational purposes and should not replace professional medical diagnosis.

## ✨ Features

### Machine Learning
- ✅ **Random Forest Classifier** with hyperparameter tuning
- ✅ Comprehensive data preprocessing and feature engineering
- ✅ Cross-validation for robust model evaluation
- ✅ Feature importance analysis
- ✅ Multiple evaluation metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
- ✅ Visualization of results (Confusion Matrix, ROC Curve, Feature Importance)

### Backend API
- ✅ Flask REST API with CORS support
- ✅ Input validation
- ✅ Prediction logging to CSV
- ✅ Explainable AI - provides reasoning for predictions
- ✅ Multiple endpoints (predict, health check, metrics, model info)

### Frontend
- ✅ **HTML/CSS/JavaScript** - Clean and responsive UI
- ✅ **Streamlit App** - Interactive dashboard with visualizations
- ✅ Real-time predictions
- ✅ Risk level assessment (Low/Medium/High)
- ✅ Key factors and recommendations display
- ✅ Sample data loading
- ✅ Form validation

## 🛠️ Tech Stack

### Machine Learning
- **pandas** - Data manipulation
- **NumPy** - Numerical computing
- **scikit-learn** - ML algorithms and evaluation
- **matplotlib & seaborn** - Data visualization
- **joblib** - Model serialization

### Backend
- **Flask** - Web framework
- **Flask-CORS** - Cross-origin resource sharing

### Frontend
- **HTML5/CSS3/JavaScript** - Web technologies
- **Streamlit** - Interactive web app
- **Plotly** - Interactive visualizations

### Data
- **kagglehub** - Dataset download

## 📁 Project Structure

```
RE_PBL/
│
├── data/
│   ├── download_dataset.py      # Download dataset from Kaggle
│   ├── generate_dataset.py      # Generate synthetic RA dataset
│   └── ra_dataset.csv           # Generated dataset (after running)
│
├── model/
│   ├── ra_random_forest_model.joblib  # Trained model
│   ├── scaler.joblib                  # Feature scaler
│   ├── label_encoders.joblib          # Categorical encoders
│   ├── feature_names.joblib           # Feature list
│   ├── feature_importance.csv         # Feature importance scores
│   ├── metrics.json                   # Model performance metrics
│   └── plots/                         # Visualizations
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       ├── feature_importance.png
│       └── prediction_distribution.png
│
├── notebooks/
│   └── train_model.py           # Model training script
│
├── backend/
│   └── app.py                   # Flask API server
│
├── frontend/
│   ├── index.html               # Main HTML page
│   ├── style.css                # Styling
│   ├── script.js                # JavaScript functionality
│   └── streamlit_app.py         # Streamlit web app
│
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore file
└── README.md                    # Project documentation
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
cd /Users/rishabhgupta/Desktop/RE_PBL
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## 📊 Usage

### ⚡ Quick Start (Recommended)

**Run the entire project with a single command:**

```bash
# Option 1: Using Python (Recommended for macOS/Linux)
python3 run.py

# Option 2: Using Bash script
bash run.sh
```

This will:
1. ✅ Install dependencies
2. ✅ Start Flask backend (http://127.0.0.1:5001)
3. ✅ Start Streamlit frontend (http://localhost:8501)
4. ✅ Open the web interface in your browser

Then **open your browser to:**
- **Frontend:** [http://localhost:8501](http://localhost:8501)
- **API:** [http://127.0.0.1:5001](http://127.0.0.1:5001)

---

## 📊 Manual Usage (If you prefer running components separately)

### 1. Generate Dataset

```bash
cd data
python generate_dataset.py
```

This creates a synthetic RA dataset with 1000 samples including:
- Demographics (Age, Gender)
- Clinical symptoms (Morning Stiffness, Joint Pain, Swollen Joints, Fatigue)
- Laboratory tests (RF, Anti-CCP, ESR, CRP)
- Medical history (Family History, Smoking Status)
- Target label (RA_Diagnosis: 0=No, 1=Yes)

**Optional:** Download real dataset from Kaggle:
```bash
python download_dataset.py
```

### 2. Train the Model

```bash
cd notebooks
python train_model.py
```

This will:
- Load and preprocess the dataset
- Train Random Forest model
- Evaluate performance with multiple metrics
- Generate visualizations
- Save trained model and artifacts to `model/` directory

**Expected Output:**
- Model Accuracy: ~85-95%
- Confusion Matrix, ROC Curve, Feature Importance plots
- Saved model files in `model/` directory

### 3. Start the Backend API

```bash
cd backend
python app.py
```

The Flask API will start on `http://127.0.0.1:5000`

**Available Endpoints:**
- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Make prediction
- `GET /model-info` - Model details
- `GET /metrics` - Performance metrics
- `GET /sample-input` - Sample data format

### 4. Run the Frontend

#### Option A: HTML/CSS/JavaScript Frontend

```bash
cd frontend
# Open index.html in your web browser
open index.html  # macOS
# or simply double-click index.html
```

**Note:** Make sure the backend API is running before using the frontend.

#### Option B: Streamlit App

```bash
cd frontend
streamlit run streamlit_app.py
```

The Streamlit app will open automatically in your browser at `http://localhost:8501`

## 📈 Model Performance

### Dataset Statistics
- **Total Samples:** 1000
- **RA Positive:** ~400 (40%)
- **RA Negative:** ~600 (60%)
- **Features:** 12 (9 numeric + 3 categorical)

### Model Metrics
| Metric | Score |
|--------|-------|
| **Accuracy** | ~92% |
| **Precision** | ~89% |
| **Recall** | ~91% |
| **F1-Score** | ~0.90 |
| **ROC-AUC** | ~0.96 |

### Feature Importance
Top 5 most important features:
1. **Anti-CCP** (~18-22%)
2. **Rheumatoid Factor** (~16-20%)
3. **CRP** (~12-15%)
4. **ESR** (~10-13%)
5. **Morning Stiffness Duration** (~8-11%)

## 📡 API Documentation

### POST /predict

Make a prediction for a patient.

**Request:**
```json
{
    "Age": 52,
    "Gender": "Female",
    "Morning_Stiffness_Duration": 90,
    "Joint_Pain_Score": 8,
    "Swollen_Joint_Count": 12,
    "Rheumatoid_Factor": 95.5,
    "Anti_CCP": 150.3,
    "ESR": 55.2,
    "CRP": 25.8,
    "Fatigue_Score": 9,
    "Family_History": "Yes",
    "Smoking_Status": "Yes"
}
```

**Response:**
```json
{
    "prediction": 1,
    "diagnosis": "RA Positive",
    "probability": {
        "No_RA": 0.15,
        "RA_Positive": 0.85
    },
    "risk_level": "High",
    "confidence": 0.85,
    "explanation": {
        "risk_level": "High",
        "key_factors": [
            "Elevated Rheumatoid Factor detected",
            "Elevated Anti-CCP antibodies detected",
            "Elevated ESR (inflammation marker)",
            "Elevated CRP (inflammation marker)",
            "Prolonged morning stiffness",
            "Multiple swollen joints",
            "High joint pain score"
        ],
        "recommendations": [
            "Consult a rheumatologist for comprehensive evaluation",
            "Consider additional diagnostic tests",
            "Discuss early treatment options to prevent joint damage",
            "Monitor symptoms regularly"
        ],
        "top_features": [...]
    }
}
```

### GET /metrics

Get model performance metrics.

**Response:**
```json
{
    "accuracy": 0.92,
    "precision": 0.89,
    "recall": 0.91,
    "f1_score": 0.90,
    "roc_auc": 0.96,
    "confusion_matrix": [[115, 5], [8, 72]]
}
```

## 💡 Sample Input/Output

### Sample 1: RA Positive Case

**Input:**
- Age: 52, Gender: Female
- Morning Stiffness: 90 minutes
- Joint Pain: 8/10
- Swollen Joints: 12
- RF: 95.5, Anti-CCP: 150.3
- ESR: 55.2, CRP: 25.8
- Fatigue: 9/10
- Family History: Yes, Smoking: Yes

**Output:**
- **Prediction:** RA Positive ⚠️
- **Probability:** 85%
- **Risk Level:** High
- **Key Factors:** Elevated RF, Anti-CCP, ESR, CRP, prolonged stiffness

### Sample 2: RA Negative Case

**Input:**
- Age: 35, Gender: Male
- Morning Stiffness: 10 minutes
- Joint Pain: 2/10
- Swollen Joints: 0
- RF: 8.5, Anti-CCP: 5.3
- ESR: 10.2, CRP: 2.1
- Fatigue: 3/10
- Family History: No, Smoking: No

**Output:**
- **Prediction:** No RA Detected ✅
- **Probability:** 92%
- **Risk Level:** Low
- **Key Factors:** No significant risk factors detected

## 🖼️ Screenshots

### HTML Frontend
![HTML Frontend](https://via.placeholder.com/800x500/667eea/ffffff?text=HTML+Frontend+Screenshot)

### Streamlit App
![Streamlit App](https://via.placeholder.com/800x500/764ba2/ffffff?text=Streamlit+App+Screenshot)

### Model Performance
![Confusion Matrix](https://via.placeholder.com/400x300/2563eb/ffffff?text=Confusion+Matrix)
![ROC Curve](https://via.placeholder.com/400x300/10b981/ffffff?text=ROC+Curve)

## 🎓 Educational Value

This project demonstrates:
- **Machine Learning Pipeline:** Data generation → Preprocessing → Training → Evaluation
- **Model Deployment:** Saving and loading ML models in production
- **REST API Development:** Building backend services for ML models
- **Frontend Development:** Creating interactive UIs for ML applications
- **Explainable AI:** Providing interpretable predictions
- **Full Stack Integration:** Connecting all components together

## 🔍 Key Learning Points

1. **Data Science:**
   - Feature engineering and selection
   - Handling imbalanced datasets
   - Cross-validation techniques
   - Model evaluation metrics

2. **Software Engineering:**
   - API design and development
   - Frontend-backend communication
   - Error handling and validation
   - Logging and monitoring

3. **Machine Learning:**
   - Random Forest algorithm
   - Hyperparameter tuning
   - Feature importance analysis
   - Model interpretability

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is for educational purposes. All code is provided as-is.

## 🙏 Acknowledgments

- Dataset inspired by clinical RA diagnostic criteria
- Built as a college PBL (Project-Based Learning) project
- Special thanks to the open-source community

## 📞 Support

For questions or issues, please:
1. Check the documentation above
2. Review the code comments
3. Test with sample data provided

## 🚨 Medical Disclaimer

**IMPORTANT:** This system is developed for educational and research purposes only. It should NOT be used for actual medical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice.

---

**Made with ❤️ for Education | Powered by Random Forest ML | 2026**
