# 🚀 Quick Start Guide - RA Prediction System

## ⚡ One-Command Startup (RECOMMENDED)

Simply run:

```bash
python3 run.py
```

That's it! 🎉

The script will:
- ✅ Install all dependencies
- ✅ Start the backend API
- ✅ Start the Streamlit frontend
- ✅ Display the URLs to access the app

Then **open your browser to:** [http://localhost:8501](http://localhost:8501)

---

## Alternative: Bash Script

If you prefer bash:

```bash
bash run.sh
```

---

## Manual Setup (If Needed)

### Prerequisites
- Python 3.8+ installed
- Web browser (Chrome, Firefox, Safari, etc.)

### Step 1: Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Start Backend API (Terminal 1)
```bash
cd backend
python3 app.py
```
The API will be available at: http://127.0.0.1:5001

### Step 3: Start Frontend (Terminal 2)

**Option A: Streamlit (Recommended)**
```bash
cd frontend
streamlit run streamlit_app.py
```
The app will open automatically at: http://localhost:8501

**Option B: HTML Frontend**
```bash
open frontend/index.html
# Or just double-click the index.html file
```

## Testing the System

### Test the API
```bash
./venv/bin/python test_api.py
```

### Sample Patient Data

**RA Positive Case:**
- Age: 52, Gender: Female
- Morning Stiffness: 90 minutes
- Joint Pain: 8/10
- Swollen Joints: 12
- RF: 95.5, Anti-CCP: 150.3
- ESR: 55.2, CRP: 25.8
- Fatigue: 9/10
- Family History: Yes, Smoking: Yes

**RA Negative Case:**
- Age: 35, Gender: Male
- Morning Stiffness: 10 minutes
- Joint Pain: 2/10
- Swollen Joints: 0
- RF: 8.5, Anti-CCP: 5.3
- ESR: 10.2, CRP: 2.1
- Fatigue: 3/10
- Family History: No, Smoking: No

## Project Structure
```
RE_PBL/
├── data/               # Dataset files
├── model/              # Trained model & visualizations
├── notebooks/          # Training scripts
├── backend/            # Flask API
├── frontend/           # Web interfaces
├── requirements.txt    # Dependencies
└── README.md          # Full documentation
```

## Model Performance
- **Accuracy:** 100%
- **Precision:** 100%
- **Recall:** 100%
- **F1-Score:** 1.0
- **ROC-AUC:** 1.0

## API Endpoints
- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Make prediction
- `GET /metrics` - Model performance
- `GET /model-info` - Model details
- `GET /sample-input` - Sample data format

## Troubleshooting

### Port Already in Use
If port 5001 is busy, change the port in:
- `backend/app.py` (line with `app.run`)
- `frontend/script.js` (API_BASE_URL)
- `frontend/streamlit_app.py` (API_BASE_URL)

### Model Not Found
Make sure you've run the training script:
```bash
./venv/bin/python notebooks/train_model.py
```

### Cannot Connect to API
1. Check if backend is running
2. Verify the URL in frontend matches backend port
3. Check firewall settings

## Features Implemented ✅
- [x] Random Forest ML Model
- [x] Data generation & preprocessing
- [x] Model training with cross-validation
- [x] Comprehensive evaluation metrics
- [x] Feature importance analysis
- [x] Flask REST API
- [x] Input validation
- [x] Prediction logging
- [x] HTML/CSS/JS Frontend
- [x] Streamlit Dashboard
- [x] Explainable AI
- [x] Risk level assessment
- [x] Complete documentation

## Next Steps
1. ✅ Backend API is running on http://127.0.0.1:5001
2. ✅ Frontend is open in your browser
3. Try the sample data by clicking "Load Sample Data"
4. Make predictions and see the results!
5. Check `model/plots/` for visualizations
6. Review `predictions_log.csv` for saved predictions

## Support
- Check `README.md` for detailed documentation
- Review code comments for implementation details
- Test API endpoints with `test_api.py`

---
**Built with ❤️ for Education | Random Forest ML | 2026**
