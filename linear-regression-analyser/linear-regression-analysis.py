import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- Page Config ---
st.set_page_config(page_title="Linear Regression Master", layout="wide")
st.title("Linear Regression: 7-Step Workflow (Python Edition)")

# ==============================================================================
# STEP 1: LOAD DATA
# ==============================================================================
st.header("Step 1: Browse Data")
data_source = st.radio("Choose Data Source:", ["Built-in Demo", "Upload CSV"], horizontal=True)

df = None

if data_source == "Built-in Demo":
    dataset_name = st.selectbox("Select Dataset:", ["Boston Housing", "Car Design (mtcars)"])
    if dataset_name == "Boston Housing":
        # Load Boston Housing from URL for simplicity
        url = 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv'
        df = pd.read_csv(url)
    else:
        # Load mtcars from URL
        url = 'https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/mtcars.csv'
        df = pd.read_csv(url, index_col=0)

elif data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

if df is not None:
    st.dataframe(df.head(50))
    st.caption(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
else:
    st.warning("Please load data to proceed.")
    st.stop()

# ==============================================================================
# STEP 2: VARIABLE SELECTION
# ==============================================================================
st.header("Step 2: Pick Variables")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

col1, col2 = st.columns(2)
with col1:
    y_var = st.selectbox("Select Response Variable (Y)", numeric_cols)
with col2:
    x_vars = st.multiselect("Select Predictor Variables (X)", [c for c in numeric_cols if c != y_var], default=[c for c in numeric_cols if c != y_var][0])

if not x_vars:
    st.error("Please select at least one predictor.")
    st.stop()

# Clean Data
df_clean = df[[y_var] + x_vars].dropna()
st.write(f"Using {len(df_clean)} rows after removing empty values.")

# ==============================================================================
# STEP 3: SPLIT DATA
# ==============================================================================
st.header("Step 3: Split Data")
col1, col2 = st.columns(2)
with col1:
    seed = st.number_input("Random Seed", value=123, step=1)
with col2:
    split_size = st.slider("Training Proportion", 0.5, 0.9, 0.7, 0.05)

if st.button("Split Data"):
    st.session_state['split_done'] = True

# Logic to maintain split state
train_df, test_df = train_test_split(df_clean, train_size=split_size, random_state=seed)

st.success(f"Training Set: {len(train_df)} rows | Testing Set: {len(test_df)} rows")

# ==============================================================================
# STEP 4: CHECK ASSUMPTIONS (Base Model)
# ==============================================================================
st.header("Step 4: Check Assumptions")
st.info("These tests run on the UNTRANSFORMED training data.")

# Fit Base Model using Statsmodels (OLS)
X_train_base = sm.add_constant(train_df[x_vars])
y_train_base = train_df[y_var]
model_base = sm.OLS(y_train_base, X_train_base).fit()
residuals_base = model_base.resid
fitted_base = model_base.fittedvalues

tab1, tab2, tab3, tab4 = st.tabs(["Normality", "Homoscedasticity", "Independence", "Multicollinearity"])

with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots()
        sns.histplot(residuals_base, kde=True, color="seagreen", ax=ax)
        ax.set_title("Residuals Histogram")
        st.pyplot(fig)
    with col_b:
        fig, ax = plt.subplots()
        sm.qqplot(residuals_base, line='45', fit=True, ax=ax)
        ax.set_title("Q-Q Plot")
        st.pyplot(fig)
    
    # Shapiro-Wilk Test
    stat, p_val = stats.shapiro(residuals_base)
    st.write(f"**Shapiro-Wilk Test:** p-value = {p_val:.4f} " + ("(Normal)" if p_val > 0.05 else "(Not Normal)"))

with tab2:
    fig, ax = plt.subplots()
    ax.scatter(fitted_base, residuals_base, alpha=0.6, color="blue")
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")
    st.pyplot(fig)
    
    # Breusch-Pagan Test
    _, p_val, _, _ = het_breuschpagan(residuals_base, X_train_base)
    st.write(f"**Breusch-Pagan Test:** p-value = {p_val:.4f} " + ("(Homoscedastic)" if p_val > 0.05 else "(Heteroscedastic)"))

with tab3:
    fig, ax = plt.subplots()
    sm.graphics.tsa.plot_acf(residuals_base, ax=ax)
    st.pyplot(fig)
    
    # Durbin-Watson
    dw = durbin_watson(residuals_base)
    st.write(f"**Durbin-Watson Statistic:** {dw:.2f} (Close to 2 is good)")

with tab4:
    # VIF
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_train_base.columns
    vif_data["VIF"] = [variance_inflation_factor(X_train_base.values, i) for i in range(len(X_train_base.columns))]
    st.dataframe(vif_data)

# ==============================================================================
# STEP 5: TRANSFORMATIONS (Box-Cox & Outliers)
# ==============================================================================
st.header("Step 5: Optimize Model")

col1, col2 = st.columns(2)
with col1:
    remove_outliers = st.checkbox("Remove Outliers (Cook's D)")
with col2:
    apply_boxcox = st.checkbox("Apply Box-Cox to Y")

# --- Logic for Optimization ---
train_final = train_df.copy()

# 1. Outlier Removal
if remove_outliers:
    influence = model_base.get_influence()
    cooks = influence.cooks_distance[0]
    cutoff = 4 / len(train_final)
    train_final = train_final[cooks < cutoff]
    st.caption(f"Removed outliers. New training size: {len(train_final)}")

# 2. Box-Cox
best_lambda = None
shift = 0

if apply_boxcox:
    y_vals = train_final[y_var]
    # Handle negative values
    if y_vals.min() <= 0:
        shift = abs(y_vals.min()) + 1
        y_vals += shift
    
    # Calculate Lambda
    best_lambda = stats.boxcox_normmax(y_vals)
    
    # Apply Transform
    train_final[y_var] = stats.boxcox(y_vals, lmbda=best_lambda)
    
    st.caption(f"Applied Box-Cox. Lambda: {best_lambda:.3f} | Shift: {shift}")
    
    # Plot Curve
    fig, ax = plt.subplots(figsize=(6, 2))
    stats.boxcox_normplot(y_vals, -2, 2, plot=ax)
    ax.axvline(best_lambda, color='red')
    st.pyplot(fig)

# ==============================================================================
# STEP 6: METRICS
# ==============================================================================
st.header("Step 6: Model Metrics")

# Fit Final Model
X_train_final = sm.add_constant(train_final[x_vars])
y_train_final = train_final[y_var]
model_final = sm.OLS(y_train_final, X_train_final).fit()

# Training Metric
r2_adj = model_final.rsquared_adj

# Testing Metric
X_test = sm.add_constant(test_df[x_vars])
preds = model_final.predict(X_test)

# Inverse Transform Predictions if Box-Cox used
if apply_boxcox and best_lambda is not None:
    # Inverse Box-Cox: (y*lambda + 1)^(1/lambda) - shift
    # Check for negative bases
    preds_inv = (preds * best_lambda + 1)
    # Avoid complex numbers
    preds_inv = np.sign(preds_inv) * np.abs(preds_inv) ** (1 / best_lambda)
    preds_inv -= shift
    mse = mean_squared_error(test_df[y_var], preds_inv)
else:
    mse = mean_squared_error(test_df[y_var], preds)

col1, col2 = st.columns(2)
col1.metric("Training Adj. R-Squared", f"{r2_adj:.4f}")
col2.metric("Testing MSE (Original Scale)", f"{mse:.4f}")

with st.expander("See Final Model Summary"):
    st.write(model_final.summary())

# ==============================================================================
# STEP 7: PREDICTION
# ==============================================================================
st.header("Step 7: Predict")

input_data = {}
cols = st.columns(3)
for i, col_name in enumerate(x_vars):
    with cols[i % 3]:
        input_data[col_name] = st.number_input(f"Value for {col_name}", value=float(df[col_name].mean()))

if st.button("Calculate Prediction"):
    # Create DF for prediction
    input_df = pd.DataFrame([input_data])
    input_df = sm.add_constant(input_df, has_constant='add')
    
    # Ensure all columns match model
    # (add_constant might behave differently on single row, so we enforce alignment)
    # This aligns the columns to match the training data structure
    model_params = model_final.params.index
    for p in model_params:
        if p not in input_df.columns:
            if p == 'const':
                input_df['const'] = 1.0
            else:
                input_df[p] = 0.0 # Should not happen if UI is correct
    
    input_df = input_df[model_params] # Reorder columns
    
    # Predict
    pred_raw = model_final.predict(input_df)[0]
    
    # Inverse Transform
    final_pred = pred_raw
    if apply_boxcox and best_lambda is not None:
        base = (pred_raw * best_lambda + 1)
        final_pred = (np.sign(base) * np.abs(base) ** (1 / best_lambda)) - shift

    st.success(f"Predicted {y_var}: {final_pred:.4f}")
    
    # Equation
    equation = f"{y_var} = {model_final.params['const']:.4f}"
    for col_name in x_vars:
        coef = model_final.params[col_name]
        sign = "+" if coef >= 0 else "-"
        equation += f" {sign} {abs(coef):.4f}*{col_name}"
    
    st.code(equation)