import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("dac_output_varied_bits.csv")

X = data[['code']]
y_vout = data['vout']
y_dnl = data['DNL']
y_inl = data['INL']

model_vout = make_pipeline(StandardScaler(), PolynomialFeatures(degree=3), RandomForestRegressor(n_estimators=200, random_state=42))
model_dnl = make_pipeline(StandardScaler(), PolynomialFeatures(degree=6), RandomForestRegressor(n_estimators=300, random_state=42))
model_inl = make_pipeline(StandardScaler(), PolynomialFeatures(degree=5), RandomForestRegressor(n_estimators=300, random_state=42))

model_vout.fit(X, y_vout)
model_dnl.fit(X, y_dnl)
model_inl.fit(X, y_inl)

pred_vout_all = model_vout.predict(X)
pred_dnl_all = model_dnl.predict(X)
pred_inl_all = model_inl.predict(X)

print(f"Vout MSE: {mean_squared_error(y_vout, pred_vout_all)}")
print(f"Vout R2: {r2_score(y_vout, pred_vout_all)}")
print(f"DNL MSE: {mean_squared_error(y_dnl, pred_dnl_all)}")
print(f"DNL R2: {r2_score(y_dnl, pred_dnl_all)}")
print(f"INL MSE: {mean_squared_error(y_inl, pred_inl_all)}")
print(f"INL R2: {r2_score(y_inl, pred_inl_all)}")

print("\nEstimated Accuracy:")
print(f"Vout: {r2_score(y_vout, pred_vout_all)*100:.2f}%")
print(f"DNL:  {max(r2_score(y_dnl, pred_dnl_all)*100, 0):.2f}%")
print(f"INL:  {r2_score(y_inl, pred_inl_all)*100:.2f}%")

try:
    new_code = int(input("\nEnter a new DAC input code to predict (e.g., 4100 or 5000): "))
    new_input = pd.DataFrame({'code': [new_code]})
    if new_code < data['code'].min() or new_code > data['code'].max():
        print(f"⚠️ Warning: Code {new_code} is outside training range ({data['code'].min()}–{data['code'].max()}).")

    pred_vout = model_vout.predict(new_input)[0]
    pred_dnl = model_dnl.predict(new_input)[0]
    pred_inl = model_inl.predict(new_input)[0]

    print("\nPrediction for new DAC input:")
    print(f"Code: {new_code}")
    print(f"Predicted Vout: {pred_vout:.6f}")
    print(f"Predicted DNL:  {pred_dnl:.6f}")
    print(f"Predicted INL:  {pred_inl:.6f}")

except ValueError:
    print("❌ Please enter a valid integer DAC code.")

plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.plot(X['code'], y_vout, 'b.', label="Actual Vout")
plt.plot(X['code'], pred_vout_all, 'r-', label="Predicted Vout")
plt.title("Vout: Actual vs Predicted")
plt.xlabel("DAC Code")
plt.ylabel("Vout (V)")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(X['code'], y_dnl, 'b.', label="Actual DNL")
plt.plot(X['code'], pred_dnl_all, 'r-', label="Predicted DNL")
plt.title("DNL: Actual vs Predicted")
plt.xlabel("DAC Code")
plt.ylabel("DNL (LSB)")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(X['code'], y_inl, 'b.', label="Actual INL")
plt.plot(X['code'], pred_inl_all, 'r-', label="Predicted INL")
plt.title("INL: Actual vs Predicted")
plt.xlabel("DAC Code")
plt.ylabel("INL (LSB)")
plt.legend()

plt.tight_layout()
plt.show()

import seaborn as sns
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
sns.residplot(x=y_vout, y=(y_vout - pred_vout), lowess=True, color='red')
plt.title("Residual Plot for Vout")
plt.xlabel("Actual Vout")
plt.ylabel("Residuals")

plt.subplot(3, 1, 2)
sns.residplot(x=y_dnl, y=(y_dnl - pred_dnl), lowess=True, color='green')
plt.title("Residual Plot for DNL")
plt.xlabel("Actual DNL")
plt.ylabel("Residuals")

plt.subplot(3, 1, 3)
sns.residplot(x=y_inl, y=(y_inl - pred_inl), lowess=True, color='blue')
plt.title("Residual Plot for INL")
plt.xlabel("Actual INL")
plt.ylabel("Residuals")

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
corr = data[['code', 'vout', 'DNL', 'INL']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()
