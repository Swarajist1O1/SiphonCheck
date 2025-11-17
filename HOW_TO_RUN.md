# 🚀 **HOW TO RUN THE DIABETES PREDICTION WEB APPLICATION**

## ✅ **Currently Running!**
Your application is **already running** at: **http://localhost:5001**

---

## 📋 **Step-by-Step Instructions**

### **1. Prerequisites Check** ✅
- ✅ Python virtual environment is set up
- ✅ All required packages are installed
- ✅ Model files are trained and ready
- ✅ Flask app is configured

### **2. Starting the Application**

#### **Option A: Direct Command (Recommended)**
```bash
cd /home/swaraj/Documents/adios2
/home/swaraj/Documents/adios2/.venv/bin/python app.py
```

#### **Option B: Using Python Module**
```bash
cd /home/swaraj/Documents/adios2
source .venv/bin/activate
python app.py
```

#### **Option C: Custom Port (if port 5000 is busy)**
```bash
cd /home/swaraj/Documents/adios2
/home/swaraj/Documents/adios2/.venv/bin/python -c "
from app import app, predictor
predictor.load_or_train_model()
app.run(debug=True, host='0.0.0.0', port=5001)
"
```

---

## 🌐 **How to Access the Application**

### **Web Interface:**
- **Primary URL**: http://localhost:5001 (currently running)
- **Alternative**: http://127.0.0.1:5001
- **Network Access**: http://10.162.231.146:5001

### **What You'll See:**
1. **Beautiful Home Page** with diabetes risk assessment form
2. **Input Fields** for all medical parameters:
   - Pregnancies, Glucose, Blood Pressure
   - Skin Thickness, Insulin, BMI
   - Diabetes Pedigree Function, Age
3. **Submit Button** to get instant predictions

### **API Endpoint:**
- **URL**: http://localhost:5001/api/predict
- **Method**: POST
- **Content-Type**: application/json

---

## 🧪 **Testing the Application**

### **Test 1: Using the Web Form**
1. Open http://localhost:5001 in your browser
2. Fill in sample values:
   - Pregnancies: 2
   - Glucose: 120
   - Blood Pressure: 70
   - Skin Thickness: 20
   - Insulin: 80
   - BMI: 25
   - Diabetes Pedigree Function: 0.5
   - Age: 30
3. Click "Predict Diabetes Risk"
4. View results with confidence scores

### **Test 2: Using API (Terminal)**
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 2,
    "Glucose": 120, 
    "BloodPressure": 70,
    "SkinThickness": 20,
    "Insulin": 80,
    "BMI": 25,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 30
  }' \
  http://localhost:5001/api/predict
```

**Expected Response:**
```json
{
  "confidence": 0.948,
  "prediction": 0,
  "probability_diabetes": 0.052,
  "probability_no_diabetes": 0.948
}
```

---

## 🎯 **What Each File Does**

### **Core Files:**
- **`app.py`** - Main Flask application (this is what you run)
- **`diabetes_model.pkl`** - Trained ML model (89.8% accuracy)
- **`diabetes_scaler.pkl`** - Feature scaling parameters
- **`diabetes_noise_synthetic.csv`** - Training dataset

### **Templates:**
- **`templates/index.html`** - Home page with input form
- **`templates/result.html`** - Results display page  
- **`templates/about.html`** - Information about the model

---

## 🔧 **Troubleshooting**

### **Problem 1: Port Already in Use**
```bash
# Check what's using port 5000
lsof -i :5000

# Kill existing processes
pkill -f "python app.py"

# Start on different port
python -c "from app import app, predictor; predictor.load_or_train_model(); app.run(port=5001)"
```

### **Problem 2: Module Not Found**
```bash
# Activate virtual environment
cd /home/swaraj/Documents/adios2
source .venv/bin/activate
pip install flask pandas scikit-learn joblib numpy

# Then run
python app.py
```

### **Problem 3: Model Not Loading**
```bash
# Check if model files exist
ls -la *.pkl *.csv

# If missing, run the training pipeline first
python advanced_data_engineering.py
```

---

## 💡 **Pro Tips**

### **Development Mode:**
- The app runs in **debug mode** by default
- Changes to code automatically reload the server
- Detailed error messages are shown

### **Production Mode:**
```python
# In app.py, change the last line to:
app.run(debug=False, host='0.0.0.0', port=5001)
```

### **Background Running:**
```bash
# Run in background
nohup python app.py > flask_app.log 2>&1 &

# Check if running
ps aux | grep "python app.py"

# Stop background process
pkill -f "python app.py"
```

---

## ✅ **Current Status**

**🟢 RUNNING**: Your application is currently active at http://localhost:5001

**🎯 Features Available:**
- ✅ Web form for diabetes risk assessment
- ✅ Real-time predictions with 89.8% accuracy  
- ✅ JSON API for external integrations
- ✅ Beautiful, responsive user interface
- ✅ Confidence scoring and risk analysis
- ✅ Printable results for medical records

**👥 Ready for users!** Just share the URL: **http://localhost:5001**

---

## 🎉 **Quick Start Summary**

```bash
# 1. Navigate to project
cd /home/swaraj/Documents/adios2

# 2. Run the application  
/home/swaraj/Documents/adios2/.venv/bin/python app.py

# 3. Open browser to
http://localhost:5001

# 4. Start predicting diabetes risk!
```

**That's it! Your diabetes prediction web application is ready to use! 🚀**
