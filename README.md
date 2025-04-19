Sure! Here's a concise version of the `README.md`:

---

# 🌾 Crop Yield Prediction App

A machine learning web app using **Streamlit** to predict crop yield (tonnes/hectare) based on environmental and agricultural factors.

---

## 📌 Features

- Input: Crop, State, Season, Year, Area, Production, Rainfall, Fertilizer, Pesticide  
- Predicts crop yield using the best trained model (XGBoost, LightGBM, or Random Forest)  
- Download predictions as CSV  
- Clean, user-friendly UI  

---

## 🧠 Training

Run `train.py` to:

- Preprocess data (imputation, encoding, scaling)  
- Train and tune models  
- Save model and preprocessing artifacts (`.pkl` files)

```bash
python train.py
```

---

## 🚀 Run App

```bash
streamlit run app.py
```

---

## 📁 Dataset Format

CSV must include:

```
Crop, Crop_Year, Season, State, Area, Production, Annual_Rainfall, Fertilizer, Pesticide, Yield
```

---

## ⚙️ Requirements

```bash
pip install -r requirements.txt
```

---

## 👤 Author

Apurva Singh

--- 

