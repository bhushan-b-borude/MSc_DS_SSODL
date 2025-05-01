import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import os
from catboost import CatBoostClassifier

st.set_page_config(layout="wide")

st.markdown('''
<style>
.main{
background-color: #FFFFE0;
}
<style>
''',
unsafe_allow_html= True)

# Markdown format change

st.sidebar.markdown('''
<style>
.main{
background-color: #ADD8E6;
}
<style>
''',
unsafe_allow_html= True)

st.write('''
# Payment Delay Prediction App

This app uses CatBoost to predict the ***Probability of Payment Delay***.

''')

# Help Section in Sidebar
st.sidebar.markdown("""
### Help
This app uses a machine learning model to predict the probability of payment delays. 
- **Step 1**: Upload a CSV file containing relevant customer data.
- **Step 2**: The app will predict the probability of payment delays for each entry.
- **Step 3**: Download the predictions as a CSV file.
- **Step 4**: Alternatively you can manually enter a single entry in 'Predict for Single Entry Section'.
""")

# # Path to sample file
# sample_file_path = '/Users/bhushanborude/Desktop/Projects/Python/Python for Machine Learning & Data Science Masterclass/FINN/Test.csv'

# # Read the file in binary mode
# with open(sample_file_path, 'rb') as f:
#     sample_bytes = f.read()

# Base directory of this script
BASE_DIR = os.path.dirname(__file__)

# Path to sample file (relative to this script)
sample_file_path = os.path.join(BASE_DIR, "Test.csv")

# Check existence and read the file in binary mode
if not os.path.exists(sample_file_path):
    st.error(f"Sample file not found at {sample_file_path}")
    st.stop()

with open(sample_file_path, "rb") as f:
    sample_bytes = f.read()

# Sidebar download button
st.sidebar.download_button(
    label="Download Sample Input File",
    data=sample_bytes,
    file_name="sample_input.csv",
    mime="text/csv"
)

st.sidebar.header('User Input Features')

uploaded_file = st.sidebar.file_uploader(' Upload your input file in CSV format', type=['csv'])

if uploaded_file is not None:
    # Load uploaded data
    input_df = pd.read_csv(uploaded_file)
    input_df['deposit'] = input_df['deposit'].apply(lambda x: 'Yes' if x in [True] else 'No')
    st.write("### Uploaded Data Preview", input_df.head())

    # Load trained model
    # with open('/Users/bhushanborude/Desktop/Projects/Python/Python for Machine Learning & Data Science MasterclasS/FINN/cat_model.pkl', 'rb') as model_file:
    #     cat_model = pickle.load(model_file)
    model_path = os.path.join(BASE_DIR, "cat_model.pkl")
    with open(model_path, 'rb') as model_file:
        cat_model = pickle.load(model_file)
    # Define categorical columns 
    categorical_cols = ['invoice_type', 'invoice_month', 'loyalty_status', 'deposit', 'due_day']

    # Ensure all required columns are present
    missing_cols = [col for col in categorical_cols if col not in input_df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
    else:
        # Predict probabilities
        probs = cat_model.predict_proba(input_df)[:, 1]
        input_df['delay_probability'] = probs.round(4)

        # Show results
        st.write("### Predictions", input_df[['delay_probability']].head())

        # Create downloadable link
        csv = input_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name='predicted_delay_probabilities.csv',
            mime='text/csv'
        )

# User Input for Single Prediction
st.sidebar.header("Predict for Single Entry")

# Sidebar inputs for prediction
invoice_type = st.sidebar.selectbox('Invoice Type', ['monthly', 'final'])
invoice_amount = st.sidebar.number_input('Invoice Amount', min_value=0.0)ß
invoice_month = st.sidebar.number_input('Invoice Month', min_value=1, max_value=12)
due_day = st.sidebar.number_input('Due Day', min_value=1, max_value=31)
deal_term = st.sidebar.number_input('Deal Term (months)', min_value=1)
deal_monthly_amount = st.sidebar.number_input('Deal Monthly Amount', min_value=0.0)
schufa_score = st.sidebar.number_input('Schufa Score', min_value=0.0)
loyalty_status = st.sidebar.selectbox('Loyalty Status', ['gold', 'silver', 'platinum'])
deposit_str = st.sidebar.selectbox('Deposit', ['Yes', 'No'])
deposit = True if deposit_str == 'Yes' else False
age = st.sidebar.number_input('Customer Age', min_value=18, max_value=100)

# Predict button for user input
if st.sidebar.button('Predict Payment Delay'):
    user_input = pd.DataFrame([{
        'invoice_type': invoice_type,
        'invoice_amount': invoice_amount,
        'invoice_month': invoice_month,
        'due_day': due_day,
        'deal_term': deal_term,
        'deal_monthly_amount': deal_monthly_amount,
        'schufa_score': schufa_score,
        'loyalty_status': loyalty_status,
        'deposit': deposit,
        'age': age,
    }])

    # Load trained model
    # with open('/Users/bhushanborude/Desktop/Projects/Python/Python for Machine Learning & Data Science MasterclasS/FINN/cat_model.pkl', 'rb') as model_file:
    #     cat_model = pickle.load(model_file)
    model_path = os.path.join(BASE_DIR, "cat_model.pkl")
    with open(model_path, 'rb') as model_file:
        cat_model = pickle.load(model_file)
    # Predict based on the input
    delay_prob = cat_model.predict_proba(user_input)[:, 1]
    st.sidebar.markdown(f"Predicted Payment Delay Probability: {delay_prob[0]:.4f}")

# Define a base path
# base_path = '/Users/bhushanborude/Desktop/Projects/Python/Python for Machine Learning & Data Science MasterclasS/FINN'
base_path = BASE_DIR
# Filenames
image_filenames = [
    'schufa_group_delay_plot.png',
    'deal_monthly_amount_group_delay_plot.png',
    'deal_term_group_delay_plot.png',
    'deposit_delay_plot.png',
    'due_day_group_delay_plot.png',
    'invoice_type_delay_plot.png',
    'invoice_amount_group_delay_plot.png',
    'loyalty_status_delay_plot.png',
    'age_group_delay_plot.png',
    'invoice_month_delay_plot.png'
]

# Combine base path with filenames
image_paths = [os.path.join(base_path, fname) for fname in image_filenames]

# Display 5x2 image grid
st.markdown("### Payment Delay Analysis Across Different Variables")

rows = [image_paths[i:i+2] for i in range(0, len(image_paths), 2)]
# image_width = 1200 
for row in rows:
    cols = st.columns(2)
    for col, img_path in zip(cols, row):
        if os.path.exists(img_path):
            image = Image.open(img_path)
            # image = image.resize((image_width, int(image_width * 0.75)))
            col.image(image, use_container_width=True, width=1200)
        else:
            col.error(f"Image not found: {os.path.basename(img_path)}")


