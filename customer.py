import streamlit as st
import pandas as pd
import numpy as np
import warnings
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import traceback

warnings.filterwarnings('ignore')

# Load Data
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("Customer Churn.csv")
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

data = load_data()

# Preprocess Data
try:
    data['Tenure'].fillna(data['Tenure'].median(), inplace=True)
    data['WarehouseToHome'].fillna(data['WarehouseToHome'].median(), inplace=True)
    data['HourSpendOnApp'].fillna(data['HourSpendOnApp'].median(), inplace=True)
    data['OrderAmountHikeFromlastYear'].fillna(data['OrderAmountHikeFromlastYear'].median(), inplace=True)
    data['CouponUsed'].fillna(data['CouponUsed'].median(), inplace=True)
    data['OrderCount'].fillna(data['OrderCount'].median(), inplace=True)
    data['DaySinceLastOrder'].fillna(data['DaySinceLastOrder'].median(), inplace=True)
    data.drop(columns=["CustomerID"], inplace=True)

    # Create new features
    data['ActivityLevel'] = data['HourSpendOnApp'] / data['Tenure'].replace(0, 1e-10)
    data['CouponUsageRate'] = data['CouponUsed'] / data['OrderCount'].replace(0, 1e-10)
    data['ComplaintRate'] = data['Complain'] / data['OrderCount'].replace(0, 1e-10)
except Exception as e:
    st.error(f"Error during preprocessing: {e}")
    st.stop()

# Features for clustering
features = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'OrderAmountHikeFromlastYear', 'CouponUsed',
            'OrderCount', 'DaySinceLastOrder', 'CashbackAmount', 'ActivityLevel','CouponUsageRate','ComplaintRate']

try:
    # Normalize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[features])
except Exception as e:
    st.error(f"Error scaling features: {e}")
    st.stop()

# Apply K-means clustering
optimal_clusters = 4
try:
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(scaled_features)
except Exception as e:
    st.error(f"Error during K-means clustering: {e}")
    st.error(traceback.format_exc())
    st.stop()

# Retention Strategies
retention_strategies = {
    0: "High-Value Loyal Customers: Offer loyalty rewards, exclusive discounts, and early access to new products.",
    1: "At-Risk Customers: Offer personalized discounts, improve customer service, and provide personalized communication.",
    2: "Churn-Prone Customers: Conduct surveys to understand pain points, offer incentives for feedback.",
    3: "New Customers: Implement onboarding programs, offer welcome discounts and special offers."
}

# Dictionary to store the original unique values and their encoded values
encoding_info_dict = {}

try:
    data['RetentionStrategy'] = data['Cluster'].map(retention_strategies)

    # Encode categorical features and store original values mapping
    categorical_features = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender', 'PreferedOrderCat', 'MaritalStatus']
    for feature in categorical_features:
        data[feature] = data[feature].astype('category')
        encoding_info_dict[feature] = dict(enumerate(data[feature].cat.categories))
        data[feature] = data[feature].cat.codes

    # Prepare data for modeling
    X = data.drop(columns=['Churn', 'RetentionStrategy'])
    y = data['Churn']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost model
    xgb_model = xgb.XGBClassifier(learning_rate=0.01, n_estimators=500, max_depth=10, gamma=1, subsample=0.8, random_state=42, enable_categorical=True)
    xgb_model.fit(X_train, y_train)
except Exception as e:
    st.error(f"Error during model training: {e}")
    st.error(traceback.format_exc())
    st.stop()

# Streamlit App
st.title("Customer Churn Prediction App")
st.write("This app predicts if a customer will churn and suggests a retention strategy.")

# Create a sidebar for categorical encoding information
st.sidebar.header("Categorical Encoding Information")
for feature in encoding_info_dict:
    st.sidebar.subheader(f"{feature}")
    st.sidebar.write(encoding_info_dict[feature])

# Create two columns
left_col, right_col = st.columns(2)

# Input form for user data in the left column
with left_col:
    st.header("Input Customer Data")
    user_data = {}
    for col in X.columns:
        if data[col].dtype == 'float64':
            if col in ['Tenure', 'WarehouseToHome', 'HourSpendOnApp']:  # Example usage of slider
                user_data[col] = st.slider(f'{col}', min_value=float(data[col].min()), max_value=float(data[col].max()), value=float(data[col].mean()))
            else:
                user_data[col] = st.number_input(f'{col}', value=float(data[col].mean()))
        elif data[col].dtype == 'int64':
            if col in ['OrderCount', 'CouponUsed']:  # Example usage of slider for int columns
                user_data[col] = st.slider(f'{col}', min_value=int(data[col].min()), max_value=int(data[col].max()), value=int(data[col].mean()))
            else:
                user_data[col] = st.number_input(f'{col}', value=int(data[col].mean()))
        else:
            unique_values = data[col].unique().tolist()
            user_data[col] = st.selectbox(f'{col}', options=unique_values)

# Convert user input to DataFrame
user_df = pd.DataFrame(user_data, index=[0])

# Add predict button
if st.button('Predict'):
    try:
        # Preprocess user data
        user_df_scaled = scaler.transform(user_df[features])

        # Predict cluster for retention strategy
        user_cluster = kmeans.predict(user_df_scaled)[0]
        retention_strategy = retention_strategies[user_cluster]

        # Predict churn
        user_churn_pred = xgb_model.predict(user_df)[0]
        churn_prob = xgb_model.predict_proba(user_df)[0][1]

        # Display results in the right column
        with right_col:
            st.header("Prediction Results")
            if user_churn_pred == 1:
                st.write("The customer is likely to churn.")
            else:
                st.write("The customer is not likely to churn.")
            st.write(f"Churn Probability: {churn_prob:.2f}")
            st.write(f"Suggested Retention Strategy: {retention_strategy}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.error(traceback.format_exc())
