# %% [markdown]
# # Libraries

# %%
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.impute import SimpleImputer

# %%


def load_housing():
    #print("Current working directory:", os.getcwd())
    df = pd.read_csv('housing.csv')
    df = df.sample(2000)
    return df

# %%
def get_model(algorithm):
    if algorithm == 'Linear Regression':
        model = LinearRegression()
    elif algorithm == 'Random Forest Regressor':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    return model

# %%
def validate_data_types(df):
    expected_types = {
        'longitude': np.float64,
        'latitude': np.float64,
        'housing_median_age': np.float64,
        'total_rooms': np.float64,
        'total_bedrooms': np.float64,
        'population': np.float64,
        'households': np.float64,
        'median_income': np.float64,
        'median_house_value': np.float64
    }
    
    for column, expected_type in expected_types.items():
        if df[column].dtype != expected_type:
            df[column] = df[column].astype(expected_type)
    return df

# %%
def train_model(df, algorithm='Linear Regression'):
    df = validate_data_types(df)  # Validate and convert data types
    X = df[['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income']]
    y = df['median_house_value']
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = get_model(algorithm)
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    # Save the model to a pickle file
    with open(f'{algorithm.lower().replace(" ", "_")}_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    return model

# %%
def evaluate_model(model, X_test, y_test):
    # Evaluate the model
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    explained_var = explained_variance_score(y_test, y_pred)

    # Calculate Adjusted R-squared
    n = len(y_test)
    k = X_test.shape[1]
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))

    # Display metrics
    st.subheader('Model Evaluation Metrics:')
    st.write(f'Mean Squared Error (MSE): {mse:.2f}')
    st.write(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
    st.write(f'Mean Absolute Error (MAE): {mae:.2f}')
    st.write(f'R-squared (R²): {r2:.4f}')
    st.write(f'Adjusted R-squared: {adj_r2:.4f}')
    st.write(f'Explained Variance Score: {explained_var:.4f}')

    # Display scatter plot and residual plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.scatter(y_test, y_pred, alpha=0.7)
    ax1.set_title('Actual vs. Predicted')
    ax1.set_xlabel('Actual')
    ax1.set_ylabel('Predicted')

    residuals = y_test - y_pred
    ax2.scatter(y_test, residuals, alpha=0.7)
    ax2.set_title('Residuals')
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Residuals')
    st.pyplot(fig)

def predict_prices(model, longitud, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income):
    input_data = [[longitud, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income]]
    predicted_prices = model.predict(input_data)
    return predicted_prices[0]

# %%
# Main Streamlit app
def main():
    st.title('California Housing Prices Prediction')

    st.header('Dataset Information')
    st.markdown("""
    ## Dataset Context
    This dataset contains information about housing prices in California. It includes various features that might influence house prices, such as location, number of rooms, population, and median income in the area.

    ## Kaggle Link
    The dataset is available on Kaggle at: [California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices)

    ## Introduction
    This Streamlit app allows you to explore and predict housing prices in California using different regression algorithms. You can adjust various features to see how they affect the predicted house price.
    """)

    # Load 
    df = load_housing()

    # Select algorithm
    algorithm = st.sidebar.selectbox('Select Regression Algorithm',
                                     ['Linear Regression', 'Random Forest Regressor'])

    # Train model
    model = train_model(df, algorithm)

    # Streamlit UI
    st.sidebar.header('User Input Features')
    longitud = st.sidebar.slider('longitude', df['longitude'].min(), df['longitude'].max(), df['longitude'].mean())
    latitude = st.sidebar.slider('latitude', df['latitude'].min(), df['latitude'].max(), df['latitude'].mean())
    housing_median_age = st.sidebar.slider('housing_median_age', df['housing_median_age'].min(), df['housing_median_age'].max(), df['housing_median_age'].mean())
    total_rooms = st.sidebar.slider('total_rooms', df['total_rooms'].min(), df['total_rooms'].max(), df['total_rooms'].mean())
    total_bedrooms = st.sidebar.slider('total_bedrooms', df['total_bedrooms'].min(), df['total_bedrooms'].max(), df['total_bedrooms'].mean())
    population = st.sidebar.slider('population', df['population'].min(), df['population'].max(), df['population'].mean())
    households = st.sidebar.slider('households', df['households'].min(), df['households'].max(), df['households'].mean())
    median_income = st.sidebar.slider('median_income', df['median_income'].min(), df['median_income'].max(), df['median_income'].mean())

    # Predict scenario
    if st.sidebar.button('Predict'):
        predicted_apparent_price = predict_prices(model, longitud, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income)
        st.sidebar.success(f'Predicted price house: {predicted_apparent_price:.2f}')
        print(f'Predicted price house: {predicted_apparent_price:.2f}')

if __name__ == '__main__':
    main()