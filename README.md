```markdown
# Global GDP Forecasting

This is the final project for the **Decision Support Systems** course.  
The goal of this project is to **forecast GDP across multiple countries** using different machine learning and statistical models, and deploy an interactive web application using **Flask**.

---

## 🚀 Project Overview
- **Topic:** Cross-country GDP forecasting  
- **Objective:** Support decision-making by predicting economic growth across nations  
- **Approach:** Compare and evaluate multiple forecasting models on panel data  
- **Deployment:** Flask-based web application with a simple "vibe-coding" style interface  

---

## 📊 Methods and Models
We experiment with several forecasting models, including:
- **Linear Regression**
- **Ridge Regression**
- **ARIMA**
- **Random Forest**
- **XGBoost**
- **Prophet**
- (Other models for comparison)

The models are trained on panel data (cross-country GDP over multiple years) to capture both temporal trends and cross-sectional differences.

---

## 🌐 Web Application
The results are deployed as a web app using **Flask**, where users can:
- Input a country and year range  
- Select a forecasting model  
- View predicted GDP and visualization charts  

---

---

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/global-gdp-forecasting.git
   cd global-gdp-forecasting
````

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Running the Flask App

```bash
python app.py
```

Then open your browser at: **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

---


---
