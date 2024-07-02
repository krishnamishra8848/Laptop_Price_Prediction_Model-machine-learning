import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load DataFrame from pickle file
@st.cache_data(hash_funcs={pd.DataFrame: lambda _: None})
def load_data(filename):
    return pd.read_pickle(filename)

# Log transformation for Storage SSD
def apply_log_transform(value):
    return np.log(value + 1)  # Adding 1 to handle zero values for log transformation

# Main function to run Streamlit app
def main():
    # Title and description
    st.title('Laptop Price Predictor')
    st.write('Enter laptop specifications to predict the price.')

    # Load DataFrame
    df = load_data('dataframe.pkl')

    # Sidebar inputs for user to enter specifications
    st.sidebar.title('Enter Specifications')

    brand_options = df['Brand'].unique()
    selected_brand = st.sidebar.selectbox('Brand', brand_options)

    model_options = df[df['Brand'] == selected_brand]['Model'].unique()
    selected_model = st.sidebar.selectbox('Model', model_options)

    processor_options = df['Processor'].unique()
    selected_processor = st.sidebar.selectbox('Processor', processor_options)

    ram_options = df['RAM'].unique()
    selected_ram = st.sidebar.selectbox('RAM', ram_options)

    manual_ssd = st.sidebar.selectbox('Select Storage SSD (GB)', ['128', '256', '512', '1024'])

    # Convert selected SSD to float and apply log transformation
    selected_ssd = float(manual_ssd)
    transformed_ssd = apply_log_transform(selected_ssd)

    size_options = df['Size'].unique()
    selected_size = st.sidebar.selectbox('Size', size_options)

    graphics_options = df['Graphics_card'].unique()
    selected_graphics = st.sidebar.selectbox('Graphics Card', graphics_options)

    # Predicting price based on user inputs
    if st.sidebar.button('Predict Price'):
        # Prepare input data for prediction
        input_data = {
            'Brand': selected_brand,
            'Model': selected_model,
            'Processor': selected_processor,
            'RAM': selected_ram,
            'Storage SSD': transformed_ssd,
            'Size': selected_size,
            'Graphics_card': selected_graphics
        }

        # Load the trained model from pickle file
        with open('random_forest_cv.pkl', 'rb') as f:
            model = pickle.load(f)

        # Make prediction
        X = pd.DataFrame([input_data])
        predicted_price = model.predict(X)[0]

        # Display prediction with larger text using HTML h1 tag
        st.subheader('Predicted Price')
        st.write(f'<h1 style="text-align:center;color:#008080;">The predicted price for the selected laptop configuration is Rs. <b>{predicted_price:.2f}</b></h1>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
