# 🚀 12-Bit Binary DAC Optimization using Machine Learning

## 📌 Project Overview
This project focuses on optimizing and predicting the performance parameters of a 12-bit Binary Digital-to-Analog Converter (DAC) using Machine Learning algorithms.

The goal is to use ML models to analyze input digital values and predict corresponding analog output behavior and linearity metrics such as Vout, DNL, and INL.

An interactive Streamlit-based web application is developed to provide real-time prediction and visualization.

---

## Web Application Preview

### Step 1: Dataset Upload Interface
![Step 1 Preview](step1_preview.png)

### Step 2: DAC Output Prediction
![Step 2 Preview](step2_preview.png)

---

## Key Features
- Upload custom DAC dataset (.csv)
- Use default dataset for quick testing
- Predict Vout, DNL, and INL values
- Display model accuracy metrics
- Interactive multi-step navigation
- Real-time prediction interface

---

##  Objectives
- Predict DAC output characteristics using ML algorithms
- Improve estimation accuracy compared to traditional analytical methods
- Provide an interactive visualization interface
- Integrate hardware concepts with AI/ML

---

##  Tech Stack
- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Streamlit

---

## Machine Learning Approach
- Data preprocessing and normalization
- Feature selection
- Model training (Linear Regression / Random Forest)
- Performance evaluation using:
  - Mean Squared Error (MSE)
  - R² Score

---

##  Results
The trained model achieves strong prediction accuracy for DAC linearity parameters.  
Vout and INL predictions show high accuracy, demonstrating the effectiveness of Machine Learning in hardware performance estimation.

---

### 1️⃣ Clone the repository

```bash
git clone https://github.com/jmaity104-ux/dac-convertor-using-ml-algorithms.git
```

### 2️⃣ Navigate into folder

```bash
cd dac-convertor-using-ml-algorithms
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the web application

```bash
streamlit run gui.py
```
