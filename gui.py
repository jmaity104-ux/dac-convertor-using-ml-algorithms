import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import seaborn as                                                       
import streamlit as st

# --- Streamlit Page Setup ---
st.set_page_config(page_title="DAC ML Predictor", layout="wide")
st.title("🔌 DAC Linearity Prediction using Machine Learning")

# --- Session State for navigation ---
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model_vout' not in st.session_state:
    st.session_state.model_vout = None

# -------------- STEP 1: UPLOAD DATASET ---------------- #
if st.session_state.step == 1:
    st.header("📂 Step 1: Upload or Use Default Dataset")
    uploaded = st.file_uploader("Upload your DAC dataset (.csv)", type=["csv"])

    if uploaded:
        data = pd.read_csv(uploaded)
        st.success("✅ Dataset uploaded successfully!")
    else:
        st.info("Using default dataset: `dac_output_varied_bits.csv`")
        data = pd.read_csv("dac_output_varied_bits.csv")

    st.dataframe(data.head())

    if st.button("➡️ Proceed to Prediction"):
        st.session_state.data = data
        st.session_state.step = 2
        st.rerun()

# -------------- STEP 2: PREDICTION ---------------- #
elif st.session_state.step == 2:
    data = st.session_state.data
    st.header("🔢 Step 2: Predict DAC Output Values")

    X = data[['code']]
    y_vout = data['vout']

    # DAC parameters
    VREF = 1.2
    LSB = VREF / (2**12 - 1)

    # Train Vout Model
    model_vout = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=3),
        RandomForestRegressor(n_estimators=200, random_state=42)
    )
    model_vout.fit(X, y_vout)
    st.session_state.model_vout = model_vout

    pred_vout_all = model_vout.predict(X)
    pred_v = np.asarray(pred_vout_all).flatten()
    codes = data['code'].values

    # Compute DNL & INL
    dnl_pred = np.zeros_like(pred_v)
    dnl_pred[0] = 0.0
    for i in range(1, len(pred_v)):
        dnl_pred[i] = ((pred_v[i] - pred_v[i-1]) / LSB) - 1.0

    vout_ideal = (codes / (2**12 - 1)) * VREF
    inl_pred = (pred_v - vout_ideal) / LSB

    # Accuracy
    mse_vout = mean_squared_error(y_vout, pred_vout_all)
    r2_vout = r2_score(y_vout, pred_vout_all)
    mse_dnl = mean_squared_error(data['DNL'], dnl_pred)
    r2_dnl = r2_score(data['DNL'], dnl_pred)
    mse_inl = mean_squared_error(data['INL'], inl_pred)
    r2_inl = r2_score(data['INL'], inl_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("Vout Accuracy", f"{r2_vout*100:.2f}%")
    col2.metric("DNL Accuracy", f"{max(r2_dnl*100,0):.2f}%")
    col3.metric("INL Accuracy", f"{r2_inl*100:.2f}%")

    new_code = st.number_input("Enter a DAC Code to Predict (0–8191)", min_value=0, max_value=8191, value=2500)
    if st.button("Predict"):
        new_input = pd.DataFrame({'code': [new_code]})
        pred_vout = model_vout.predict(new_input)[0]

        prev_code = max(new_code - 1, 0)
        v_prev = model_vout.predict(pd.DataFrame({'code': [prev_code]}))[0]

        pred_dnl = ((pred_vout - v_prev) / LSB) - 1.0
        ideal_vout = (new_code / (2**12 - 1)) * VREF
        pred_inl = (pred_vout - ideal_vout) / LSB

        st.success(f"### Prediction for Code {new_code}")
        st.write(f"**Predicted Vout:** {pred_vout:.6f} V")
        st.write(f"**Predicted DNL:** {pred_dnl:.6f} LSB")
        st.write(f"**Predicted INL:** {pred_inl:.6f} LSB")

    if st.button("➡️ Go to Graphs"):
        st.session_state.pred_vout_all = pred_vout_all
        st.session_state.dnl_pred = dnl_pred
        st.session_state.inl_pred = inl_pred
        st.session_state.step = 3
        st.rerun()

# -------------- STEP 3: GRAPH SELECTION ---------------- #
elif st.session_state.step == 3:
    data = st.session_state.data
    pred_vout_all = st.session_state.pred_vout_all
    dnl_pred = st.session_state.dnl_pred
    inl_pred = st.session_state.inl_pred
    X = data[['code']]
    y_vout = data['vout']

    st.header("📈 Step 3: Visualize Results")

    graph_choice = st.selectbox(
        "Select which graph to display:",
        ["Vout (Actual vs Predicted)", "DNL (Actual vs Predicted)", "INL (Actual vs Predicted)", 
         "Residual Plot", "Correlation Heatmap"]
    )

    if graph_choice == "Vout (Actual vs Predicted)":
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(X['code'], y_vout, 'b.', label="Actual Vout")
        ax.plot(X['code'], pred_vout_all, 'r-', label="Predicted Vout")
        ax.set_title("Vout: Actual vs Predicted")
        ax.set_xlabel("DAC Code"); ax.set_ylabel("Vout (V)")
        ax.legend()
        st.pyplot(fig)

    elif graph_choice == "DNL (Actual vs Predicted)":
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(X['code'], data['DNL'], 'b.', label="Actual DNL")
        ax.plot(X['code'], dnl_pred, 'r-', label="Predicted DNL")
        ax.set_title("DNL: Actual vs Predicted")
        ax.set_xlabel("DAC Code"); ax.set_ylabel("DNL (LSB)")
        ax.legend()
        st.pyplot(fig)

    elif graph_choice == "INL (Actual vs Predicted)":
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(X['code'], data['INL'], 'b.', label="Actual INL")
        ax.plot(X['code'], inl_pred, 'r-', label="Predicted INL")
        ax.set_title("INL: Actual vs Predicted")
        ax.set_xlabel("DAC Code"); ax.set_ylabel("INL (LSB)")
        ax.legend()
        st.pyplot(fig)

    elif graph_choice == "Residual Plot":
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.residplot(x=y_vout, y=(y_vout - pred_vout_all), color='red', ax=ax)
        ax.set_title("Residual Plot for Vout")
        ax.set_xlabel("Actual Vout"); ax.set_ylabel("Residuals")
        st.pyplot(fig)

    elif graph_choice == "Correlation Heatmap":
        corr = pd.DataFrame({
            'code': data['code'],
            'vout_pred': pred_vout_all,
            'dnl_pred': dnl_pred,
            'inl_pred': inl_pred
        }).corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title("Feature Correlation Heatmap")
        st.pyplot(fig)

    if st.button("🔙 Back to Prediction"):
        st.session_state.step = 2
        st.rerun()

st.markdown("---")
st.markdown("**Developed by:** Jaydeb Maity")
#  py -m streamlit run gui.py