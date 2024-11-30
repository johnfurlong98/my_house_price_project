import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox
import pickle

# Get the directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set up page configuration with custom theme
st.set_page_config(
    page_title="House Price Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load models and data
def load_data():
    data_path = os.path.join(BASE_DIR, 'data', 'house_prices_records.csv')
    inherited_houses_path = os.path.join(BASE_DIR, 'data', 'inherited_houses.csv')
    data = pd.read_csv(data_path)
    inherited_houses = pd.read_csv(inherited_houses_path)
    return data, inherited_houses

def load_models():
    models_dir = os.path.join(BASE_DIR, 'jupyter_notebooks', 'models')
    models = {}
    # Load all models
    model_files = {
        'Linear Regression': 'linear_regression_model.joblib',
        'Ridge Regression': 'ridge_regression_model.joblib',
        'ElasticNet': 'elasticnet_model.joblib',
        'Lasso Regression': 'lasso_regression_model.joblib',
        'Gradient Boosting': 'gradient_boosting_model.joblib',
        'Random Forest': 'random_forest_model.joblib',
        'XGBoost': 'xgboost_model.joblib'
    }
    for name, filename in model_files.items():
        model_path = os.path.join(models_dir, filename)
        models[name] = joblib.load(model_path)

    scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
    selected_features = pickle.load(open(os.path.join(models_dir, 'selected_features.pkl'), 'rb'))
    skewed_features = pickle.load(open(os.path.join(models_dir, 'skewed_features.pkl'), 'rb'))
    lam_dict = pickle.load(open(os.path.join(models_dir, 'lam_dict.pkl'), 'rb'))
    feature_importances = pd.read_csv(os.path.join(models_dir, 'feature_importances.csv'))
    return models, scaler, selected_features, skewed_features, lam_dict, feature_importances

# Load data
data, inherited_houses = load_data()

# Load models and related data
models, scaler, selected_features, skewed_features, lam_dict, feature_importances = load_models()

# Define feature engineering function
def feature_engineering(df):
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['Qual_TotalSF'] = df['OverallQual'] * df['TotalSF']
    return df

# Define preprocessing function
def preprocess_data(df):
    df_processed = df.copy()
    
    # Map full words back to codes
    user_to_model_mappings = {
        'BsmtFinType1': {
            'No Basement': 'None',
            'Unfinished': 'Unf',
            'Low Quality': 'LwQ',
            'Rec Room': 'Rec',
            'Basement Living Quarters': 'BLQ',
            'Average Living Quarters': 'ALQ',
            'Good Living Quarters': 'GLQ'
        },
        'BsmtExposure': {
            'No Basement': 'None',
            'No Exposure': 'No',
            'Minimum Exposure': 'Mn',
            'Average Exposure': 'Av',
            'Good Exposure': 'Gd'
        },
        'GarageFinish': {
            'No Garage': 'None',
            'Unfinished': 'Unf',
            'Rough Finished': 'RFn',
            'Finished': 'Fin'
        },
        'KitchenQual': {
            'Poor': 'Po',
            'Fair': 'Fa',
            'Typical/Average': 'TA',
            'Good': 'Gd',
            'Excellent': 'Ex'
        }
    }
    for col, mapping in user_to_model_mappings.items():
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].map(mapping)

    # Handle missing values
    zero_fill_features = ['2ndFlrSF', 'EnclosedPorch', 'MasVnrArea', 'WoodDeckSF',
                          'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', 'BsmtUnfSF']
    for feature in zero_fill_features:
        if feature in df_processed.columns:
            df_processed[feature] = df_processed[feature].fillna(0)

    # Fill categorical features
    categorical_mode_fill = {
        'BsmtFinType1': 'None',
        'GarageFinish': 'Unf',
        'BsmtExposure': 'No',
        'KitchenQual': 'TA'
    }
    for feature, value in categorical_mode_fill.items():
        if feature in df_processed.columns:
            df_processed[feature] = df_processed[feature].fillna(value)

    # Fill numerical features
    numerical_median_fill = ['BedroomAbvGr', 'GarageYrBlt', 'LotFrontage', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd']
    for feature in numerical_median_fill:
        if feature in df_processed.columns:
            df_processed[feature] = df_processed[feature].fillna(data[feature].median())

    # Encode categorical features
    ordinal_mappings = {
        'BsmtFinType1': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
        'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'BsmtExposure': {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
        'GarageFinish': {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
    }
    for col, mapping in ordinal_mappings.items():
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].map(mapping)

    # Feature engineering
    df_processed = feature_engineering(df_processed)

    # Transform skewed features
    for feat in skewed_features:
        if feat in df_processed.columns:
            if (df_processed[feat] <= 0).any():
                df_processed[feat] = np.log1p(df_processed[feat])
            else:
                lam = lam_dict.get(feat)
                if lam is not None:
                    try:
                        df_processed[feat] = boxcox(df_processed[feat], lam)
                    except ValueError:
                        df_processed[feat] = np.log1p(df_processed[feat])
                else:
                    df_processed[feat] = np.log1p(df_processed[feat])

    return df_processed

# Preprocess the data
data = preprocess_data(data)

# Metadata for features (from the provided metadata)
feature_metadata = {
    '1stFlrSF': 'First Floor square feet',
    '2ndFlrSF': 'Second floor square feet',
    'BedroomAbvGr': 'Bedrooms above grade (0 - 8)',
    'BsmtExposure': 'Walkout or garden level walls',
    'BsmtFinType1': 'Rating of basement finished area',
    'BsmtFinSF1': 'Type 1 finished square feet',
    'BsmtUnfSF': 'Unfinished basement area',
    'TotalBsmtSF': 'Total basement area',
    'GarageArea': 'Garage size in square feet',
    'GarageFinish': 'Garage interior finish',
    'GarageYrBlt': 'Year garage was built',
    'GrLivArea': 'Above grade living area',
    'KitchenQual': 'Kitchen quality',
    'LotArea': 'Lot size in square feet',
    'LotFrontage': 'Linear feet of street connected to property',
    'MasVnrArea': 'Masonry veneer area',
    'EnclosedPorch': 'Enclosed porch area',
    'OpenPorchSF': 'Open porch area',
    'OverallCond': 'Overall condition rating (1 - 10)',
    'OverallQual': 'Overall material and finish rating (1 - 10)',
    'WoodDeckSF': 'Wood deck area',
    'YearBuilt': 'Original construction date',
    'YearRemodAdd': 'Remodel date',
    'TotalSF': 'Total square feet of house (including basement)',
    'Qual_TotalSF': 'Product of OverallQual and TotalSF'
}

# Define feature input details for the user input form
feature_input_details = {
    'OverallQual': {
        'input_type': 'slider',
        'label': 'Overall Quality (1-10)',
        'min_value': 1,
        'max_value': 10,
        'value': 5,
        'step': 1,
        'help_text': feature_metadata['OverallQual']
    },
    'OverallCond': {
        'input_type': 'slider',
        'label': 'Overall Condition (1-10)',
        'min_value': 1,
        'max_value': 10,
        'value': 5,
        'step': 1,
        'help_text': feature_metadata['OverallCond']
    },
    'YearBuilt': {
        'input_type': 'slider',
        'label': 'Year Built',
        'min_value': 1872,
        'max_value': 2024,
        'value': 1975,
        'step': 1,
        'help_text': feature_metadata['YearBuilt']
    },
    'YearRemodAdd': {
        'input_type': 'slider',
        'label': 'Year Remodeled',
        'min_value': 1950,
        'max_value': 2024,
        'value': 1997,
        'step': 1,
        'help_text': feature_metadata['YearRemodAdd']
    },
    'GrLivArea': {
        'input_type': 'number_input',
        'label': 'Above Grade Living Area (sq ft)',
        'min_value': 0,
        'max_value': 10000,
        'value': 750,
        'step': 1,
        'help_text': feature_metadata['GrLivArea']
    },
    '1stFlrSF': {
        'input_type': 'number_input',
        'label': 'First Floor Area (sq ft)',
        'min_value': 0,
        'max_value': 10000,
        'value': 600,
        'step': 1,
        'help_text': feature_metadata['1stFlrSF']
    },
    '2ndFlrSF': {
        'input_type': 'number_input',
        'label': 'Second Floor Area (sq ft)',
        'min_value': 0,
        'max_value': 10000,
        'value': 0,
        'step': 1,
        'help_text': feature_metadata['2ndFlrSF']
    },
    'TotalBsmtSF': {
        'input_type': 'number_input',
        'label': 'Total Basement Area (sq ft)',
        'min_value': 0,
        'max_value': 10000,
        'value': 0,
        'step': 1,
        'help_text': feature_metadata['TotalBsmtSF']
    },
    'LotArea': {
        'input_type': 'number_input',
        'label': 'Lot Area (sq ft)',
        'min_value': 0,
        'max_value': 10000,
        'value': 1300,
        'step': 1,
        'help_text': feature_metadata['LotArea']
    },
    'LotFrontage': {
        'input_type': 'number_input',
        'label': 'Lot Frontage (linear ft)',
        'min_value': 0,
        'max_value': 1000,
        'value': 25,
        'step': 1,
        'help_text': feature_metadata['LotFrontage']
    },
    'BsmtFinType1': {
        'input_type': 'selectbox',
        'label': 'Basement Finish Type',
        'options': [
            'No Basement',
            'Unfinished',
            'Low Quality',
            'Rec Room',
            'Basement Living Quarters',
            'Average Living Quarters',
            'Good Living Quarters'
        ],
        'help_text': feature_metadata['BsmtFinType1']
    },
    'BsmtExposure': {
        'input_type': 'selectbox',
        'label': 'Basement Exposure',
        'options': [
            'No Basement',
            'No Exposure',
            'Minimum Exposure',
            'Average Exposure',
            'Good Exposure'
        ],
        'help_text': feature_metadata['BsmtExposure']
    },
    'BsmtFinSF1': {
        'input_type': 'number_input',
        'label': 'Finished Basement Area (sq ft)',
        'min_value': 0,
        'max_value': 10000,
        'value': 0,
        'step': 1,
        'help_text': feature_metadata['BsmtFinSF1']
    },
    'BsmtUnfSF': {
        'input_type': 'number_input',
        'label': 'Unfinished Basement Area (sq ft)',
        'min_value': 0,
        'max_value': 10000,
        'value': 0,
        'step': 1,
        'help_text': feature_metadata['BsmtUnfSF']
    },
    'GarageFinish': {
        'input_type': 'selectbox',
        'label': 'Garage Finish',
        'options': [
            'No Garage',
            'Unfinished',
            'Rough Finished',
            'Finished'
        ],
        'help_text': feature_metadata['GarageFinish']
    },
    'GarageYrBlt': {
        'input_type': 'slider',
        'label': 'Garage Year Built',
        'min_value': 1900,
        'max_value': 2024,
        'value': 1990,
        'step': 1,
        'help_text': feature_metadata['GarageYrBlt']
    },
    'GarageArea': {
        'input_type': 'number_input',
        'label': 'Garage Area (sq ft)',
        'min_value': 0,
        'max_value': 10000,
        'value': 50,
        'step': 1,
        'help_text': feature_metadata['GarageArea']
    },
    'WoodDeckSF': {
        'input_type': 'number_input',
        'label': 'Wood Deck Area (sq ft)',
        'min_value': 0,
        'max_value': 10000,
        'value': 0,
        'step': 1,
        'help_text': feature_metadata['WoodDeckSF']
    },
    'OpenPorchSF': {
        'input_type': 'number_input',
        'label': 'Open Porch Area (sq ft)',
        'min_value': 0,
        'max_value': 10000,
        'value': int(data['OpenPorchSF'].median()),
        'step': 1,
        'help_text': feature_metadata['OpenPorchSF']
    },
    'EnclosedPorch': {
        'input_type': 'number_input',
        'label': 'Enclosed Porch Area (sq ft)',
        'min_value': 0,
        'max_value': 10000,
        'value': int(data['EnclosedPorch'].median()),
        'step': 1,
        'help_text': feature_metadata['EnclosedPorch']
    },
    'BedroomAbvGr': {
        'input_type': 'slider',
        'label': 'Bedrooms Above Grade',
        'min_value': 0,
        'max_value': 8,
        'value': int(data['BedroomAbvGr'].median()),
        'step': 1,
        'help_text': feature_metadata['BedroomAbvGr']
    },
    'KitchenQual': {
        'input_type': 'selectbox',
        'label': 'Kitchen Quality',
        'options': [
            'Poor',
            'Fair',
            'Typical/Average',
            'Good',
            'Excellent'
        ],
        'help_text': feature_metadata['KitchenQual']
    },
    'MasVnrArea': {
        'input_type': 'number_input',
        'label': 'Masonry Veneer Area (sq ft)',
        'min_value': 0,
        'max_value': 10000,
        'value': int(data['MasVnrArea'].median()),
        'step': 1,
        'help_text': feature_metadata['MasVnrArea']
    },
}

# Apply custom CSS for enhanced UI (optional)
st.markdown(
    """
    <style>
    /* Custom CSS for the tabs and form */
    .stTabs [role="tablist"] {
        justify-content: center;
    }
    .st-form {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create tabs for navigation
tabs = ["Project Summary", "Feature Correlations", "House Price Predictions", "Project Hypotheses", "Model Performance"]
tab1, tab2, tab3, tab4, tab5 = st.tabs(tabs)

# Project Summary Page
with tab1:
    st.title("House Price Prediction Dashboard")
    st.write("""
    ## Project Summary

    Welcome to the House Price Prediction Dashboard. This project aims to build a predictive model to estimate the sale prices of houses based on various features. By analyzing the data and developing robust models, we provide insights into the key factors that influence house prices.

    *Key Objectives:*

    - *Data Analysis and Preprocessing:* Understand and prepare the data for modeling.
    - *Feature Engineering:* Create new features to improve model performance.
    - *Model Development:* Train and evaluate multiple regression models.
    - *Deployment:* Develop an interactive dashboard for predictions and insights.

    *Instructions:*

    - Use the tabs at the top to navigate between different sections.
    - Explore data correlations, make predictions, and understand the model performance.
    """)

# Feature Correlations Page
with tab2:
    st.title("Feature Correlations")
    st.write("""
    Understanding the relationships between different features and the sale price is crucial for building an effective predictive model.
    """)

    # Compute correlation matrix
    corr_matrix = data.corr()
    top_corr_features = corr_matrix.index[abs(corr_matrix['SalePrice']) > 0.5]

    # Plot correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(data[top_corr_features].corr(), annot=True, cmap='RdBu')
    plt.title('Correlation Heatmap of Top Features', fontsize=16)
    st.pyplot(plt)

    st.write("""
    The heatmap above shows the correlations among the top features and the sale price. Features like OverallQual, GrLivArea, and TotalSF have strong positive correlations with SalePrice.
    """)

    # Additional visualization: Pairplot with top features
    st.write("### Pairplot of Top Correlated Features")
    # Select top 5 features excluding 'SalePrice'
    top_features = top_corr_features.drop('SalePrice').tolist()[:5]
    sns.set(style="ticks")
    pairplot_fig = sns.pairplot(data[top_features + ['SalePrice']], diag_kind='kde', height=2.5)
    st.pyplot(pairplot_fig)

    st.write("""
    The pairplot displays pairwise relationships between the top correlated features and the sale price. It helps visualize potential linear relationships and distributions.
    """)

# House Price Predictions Page
with tab3:
    st.title("House Price Predictions")

    # Inherited Houses Predictions
    st.header("Inherited Houses")
    st.write("Below are the predicted sale prices for the inherited houses based on the best-performing model.")

    # Preprocess and predict for inherited houses
    inherited_processed = preprocess_data(inherited_houses)
    inherited_scaled = scaler.transform(inherited_processed[selected_features])
    best_model_name = 'XGBoost'  # Assuming XGBoost is the best model
    selected_model = models[best_model_name]
    predictions_log = selected_model.predict(inherited_scaled)
    predictions_actual = np.expm1(predictions_log)
    predictions_actual[predictions_actual < 0] = 0  # Handle negative predictions

    # Add predictions to the processed DataFrame
    inherited_processed['Predicted SalePrice'] = predictions_actual

    # Display the DataFrame with the selected features
    st.dataframe(inherited_processed[['Predicted SalePrice'] + selected_features])
    total_predicted_price = predictions_actual.sum()
    st.success(f"The total predicted sale price for all inherited houses is *${total_predicted_price:,.2f}*.")

    # Real-Time Prediction
    st.header("Real-Time House Price Prediction")
    st.write("Input house attributes to predict the sale price using the best-performing model.")

    def user_input_features():
        input_data = {}
        with st.form(key='house_features'):
            st.write("### Enter House Attributes")
            # Group features into sections
            feature_groups = {
                'General': ['OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd'],
                'Area': ['GrLivArea', '1stFlrSF', '2ndFlrSF', 'TotalBsmtSF', 'LotArea', 'LotFrontage'],
                'Basement': ['BsmtFinType1', 'BsmtExposure', 'BsmtFinSF1', 'BsmtUnfSF'],
                'Garage': ['GarageFinish', 'GarageYrBlt', 'GarageArea'],
                'Porch/Deck': ['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch'],
                'Other': ['BedroomAbvGr', 'KitchenQual', 'MasVnrArea'],
            }

            for group_name, features in feature_groups.items():
                st.subheader(group_name)
                cols = st.columns(2)
                idx = 0
                for feature in features:
                    if feature in feature_input_details:
                        details = feature_input_details[feature]
                        input_type = details['input_type']
                        label = details['label']
                        help_text = details['help_text']
                        with cols[idx % 2]:
                            if input_type == 'number_input':
                                input_data[feature] = st.number_input(
                                    label,
                                    min_value=details['min_value'],
                                    max_value=details['max_value'],
                                    value=details['value'],
                                    step=details['step'],
                                    help=help_text
                                )
                            elif input_type == 'slider':
                                input_data[feature] = st.slider(
                                    label,
                                    min_value=details['min_value'],
                                    max_value=details['max_value'],
                                    value=details['value'],
                                    step=details['step'],
                                    help=help_text
                                )
                            elif input_type == 'selectbox':
                                input_data[feature] = st.selectbox(
                                    label,
                                    options=details['options'],
                                    index=details.get('index', 0),
                                    help=help_text
                                )
                        idx += 1  # Increment idx to switch columns

            submit_button = st.form_submit_button(label='Predict Sale Price')

        if submit_button:
            input_df = pd.DataFrame(input_data, index=[0])
            # Calculate engineered features
            input_df = feature_engineering(input_df)
            return input_df
        else:
            return None

    user_input = user_input_features()
    if user_input is not None:
        user_processed = preprocess_data(user_input)
        user_scaled = scaler.transform(user_processed[selected_features])
        user_pred_log = selected_model.predict(user_scaled)
        user_pred_actual = np.expm1(user_pred_log)
        user_pred_actual[user_pred_actual < 0] = 0  # Handle negative predictions
        st.success(f"The predicted sale price is *${user_pred_actual[0]:,.2f}*.")

# Project Hypotheses Page
with tab4:
    st.title("Project Hypotheses")
    st.write("""
    ## Hypothesis Validation

    *Hypothesis 1:* Higher overall quality of the house leads to a higher sale price.

    - *Validation:* The OverallQual feature shows a strong positive correlation with the sale price, confirming this hypothesis.

    *Hypothesis 2:* Larger living areas result in higher sale prices.

    - *Validation:* Features like GrLivArea and TotalSF have high correlations with the sale price, supporting this hypothesis.

    *Hypothesis 3:* Recent renovations positively impact the sale price.

    - *Validation:* The YearRemodAdd feature correlates with the sale price, indicating that more recent remodels can increase the house value.
    """)

    # Visualization for Hypotheses
    st.write("### Visualization of Hypotheses")

    # OverallQual vs SalePrice
    st.write("#### SalePrice vs OverallQual")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='OverallQual', y='SalePrice', data=data, palette='Set2')
    plt.title('SalePrice vs OverallQual', fontsize=16)
    plt.xlabel('Overall Quality', fontsize=12)
    plt.ylabel('Sale Price', fontsize=12)
    st.pyplot(plt)

    # TotalSF vs SalePrice
    st.write("#### SalePrice vs TotalSF")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='TotalSF', y='SalePrice', data=data, hue='OverallQual', palette='coolwarm')
    plt.title('SalePrice vs TotalSF', fontsize=16)
    plt.xlabel('Total Square Footage', fontsize=12)
    plt.ylabel('Sale Price', fontsize=12)
    st.pyplot(plt)

    # YearRemodAdd vs SalePrice
    st.write("#### SalePrice vs YearRemodAdd")
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='YearRemodAdd', y='SalePrice', data=data, ci=None, color='green')
    plt.title('SalePrice vs Year Remodeled', fontsize=16)
    plt.xlabel('Year Remodeled', fontsize=12)
    plt.ylabel('Average Sale Price', fontsize=12)
    st.pyplot(plt)

# Model Performance Page
with tab5:
    st.title("Model Performance")
    st.header("Performance Metrics")
    results_df_path = os.path.join(BASE_DIR, 'jupyter_notebooks', 'models', 'model_evaluation.csv')
    results_df = pd.read_csv(results_df_path)
    st.dataframe(results_df.style.format({'MAE': '{:,.2f}', 'RMSE': '{:,.2f}', 'R² Score': '{:.4f}'}))

    # Determine best model
    best_model_name = results_df.sort_values('RMSE').iloc[0]['Model']
    st.write(f"Best performing model is *{best_model_name}* based on RMSE.")

    st.write("""
    The table above presents the performance metrics of various regression models. The best-performing model outperforms others with the lowest MAE and RMSE, and the highest R² Score.
    """)

    st.header("Detailed Pipeline Explanation")
    st.write("""
    ### 1. Data Collection and Understanding
    - *Datasets Used:*
      - *Historical House Sale Data:* Contains features and sale prices of houses.
      - *Inherited Houses Data:* Contains features of houses for which sale prices need to be predicted.
    - *Exploratory Data Analysis (EDA):*
      - Assessed data shapes, types, and initial statistics.
      - Identified potential relationships and patterns.

    ### 2. Data Cleaning
    - *Handling Missing Values:*
      - *Numerical Features:* Filled missing values with zeros or the median of the feature.
      - *Categorical Features:* Filled missing values with the mode or a default category.
      - *Verification:* Confirmed that no missing values remained after imputation.

    ### 3. Feature Engineering
    - *Categorical Encoding:*
      - Applied ordinal encoding to convert categorical features into numerical values based on domain knowledge.
    - *Creation of New Features:*
      - *TotalSF:* Combined total square footage of the house, including basement and above-ground areas.
      - *Qual_TotalSF:* Product of OverallQual and TotalSF to capture the combined effect of size and quality.

    ### 4. Feature Transformation
    - *Addressing Skewness:*
      - Identified skewed features using skewness metrics.
      - Applied log transformation or Box-Cox transformation to normalize distributions.

    ### 5. Feature Selection
    - *Random Forest Feature Importances:*
      - Trained a Random Forest model to assess feature importances.
      - Selected top features contributing most to the model's predictive power.

    ### 6. Data Scaling
    - *Standardization:*
      - Used StandardScaler to standardize features.
      - Ensured that features have a mean of 0 and a standard deviation of 1 for optimal model performance.

    ### 7. Model Training
    - *Algorithms Used:*
      - Linear Regression, Ridge Regression, Lasso Regression, ElasticNet, Random Forest, Gradient Boosting, XGBoost.
    - *Hyperparameter Tuning:*
      - Adjusted parameters using techniques like cross-validation to prevent overfitting and improve generalization.

    ### 8. Model Evaluation
    - *Performance Metrics:*
      - *Mean Absolute Error (MAE):* Measures average magnitude of errors.
      - *Root Mean Squared Error (RMSE):* Penalizes larger errors more than MAE.
      - *R² Score:* Indicates the proportion of variance explained by the model.
    - *Best Model Selection:*
      - Selected the model with the lowest RMSE and highest R² Score.

    ### 9. Deployment
    - *Interactive Dashboard:*
      - Developed using Streamlit for real-time interaction.
      - Allows users to input house features and obtain immediate sale price predictions.
      - Provides visualizations and insights into model performance and data relationships.
    """)

    st.header("Feature Importances")
    # Display feature importances from the Random Forest model
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importances.sort_values(by='Importance', ascending=False))
    plt.title('Feature Importances from Random Forest', fontsize=16)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    st.pyplot(plt)

    st.write("""
    The bar chart illustrates the relative importance of each feature in predicting the sale price. Features like GrLivArea, TotalSF, and OverallQual are among the most significant.
    """)

    st.header("Actual vs Predicted Prices")
    selected_model = models[best_model_name]
    train_test_data_path = os.path.join(BASE_DIR, 'jupyter_notebooks', 'models', 'train_test_data.joblib')
    X_train, X_test, y_train, y_test = joblib.load(train_test_data_path)
    y_pred_log = selected_model.predict(X_test)
    y_pred_actual = np.expm1(y_pred_log)
    y_pred_actual[y_pred_actual < 0] = 0  # Handle negative predictions
    y_test_actual = np.expm1(y_test)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test_actual, y=y_pred_actual, color='purple')
    plt.xlabel('Actual Sale Price', fontsize=12)
    plt.ylabel('Predicted Sale Price', fontsize=12)
    plt.title('Actual vs Predicted Sale Prices', fontsize=16)
    plt.plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], 'r--')
    st.pyplot(plt)

    st.write("""
    The scatter plot compares the actual sale prices with the predicted sale prices. The red dashed line represents perfect predictions. Most points are close to this line, indicating good model performance.
    """)

    st.header("Residual Analysis")
    residuals = y_test_actual - y_pred_actual
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color='coral')
    plt.title('Residuals Distribution', fontsize=16)
    plt.xlabel('Residuals', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    st.pyplot(plt)

    st.write("""
    The residuals are centered around zero and approximately normally distributed, suggesting that the model's errors are random and unbiased.
    """)
