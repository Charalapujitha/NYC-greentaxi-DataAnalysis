
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title("📈 Multiple Linear Regression Analysis")

# Load data
st.header("Step 1: Upload Dataset")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Dataset")
    st.write(df.head())

    st.subheader("Select Features and Target")
    target = st.selectbox("Select target variable (Y):", df.columns)
    features = st.multiselect("Select feature variables (X):", [col for col in df.columns if col != target])

    if features:
        X = df[features]
        y = df[target]

        st.write("Shape of X:", X.shape)
        st.write("Shape of y:", y.shape)

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model Training
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Display Results
        st.header("Step 2: Model Evaluation")
        st.write("Coefficients:", model.coef_)
        st.write("Intercept:", model.intercept_)
        st.write("R^2 Score:", r2_score(y_test, y_pred))
        st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))

        # Plot Actual vs Predicted
        st.subheader("Actual vs Predicted Plot")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.7)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        st.pyplot(fig)
