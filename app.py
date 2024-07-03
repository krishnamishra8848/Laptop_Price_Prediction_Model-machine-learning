import streamlit as st
import pandas as pd
import numpy as np
import requests

# Load DataFrame from pickle file
@st.cache(hash_funcs={pd.DataFrame: lambda _: None})
def load_data(filename):
    return pd.read_pickle(filename)

# Log transformation for Storage SSD
def apply_log_transform(value):
    return np.log(value + 1)  # Adding 1 to handle zero values for log transformation

# Translate Yes/No to Nepali
def translate_to_nepali(value):
    if value == 1:
        return "हो (Yo)"
    else:
        return ""

# Function to make API request and get prediction
def get_prediction(input_data):
    url = 'https://laptop-price-prediction-model-nepal.onrender.com/predict'  # Replace with your Render app URL
    response = requests.post(url, json=input_data)
    if response.status_code == 200:
        return response.json()['predicted_price']
    else:
        st.error(f"Error: {response.status_code} - {response.json()['message']}")

def main():
    # Title and description
    st.title('Laptop Price Predictor Nepal')

    # Load DataFrame
    df = load_data('dataframe.pkl')

    col1, col2 = st.columns([1, 3])  # Adjusting column widths

    with col1:
        brand_options = df['Brand'].unique()
        selected_brand = st.selectbox('ब्रान्ड', brand_options)

        model_options = df[df['Brand'] == selected_brand]['Model'].unique()
        selected_model = st.selectbox('मोडेल', model_options)

        processor_options = df['Processor'].unique()
        selected_processor = st.selectbox('प्रोसेसर', processor_options)

        ram_options = df['RAM'].unique()
        selected_ram = st.selectbox('RAM', ram_options)

    with col2:
        manual_ssd = st.selectbox('स्टोरेज SSD (जीबीमा चयन गर्नुहोस्)', ['128', '256', '512', '1024'])

        # Convert selected SSD to float and apply log transformation
        selected_ssd = float(manual_ssd)
        transformed_ssd = apply_log_transform(selected_ssd)

        size_options = df['Size'].unique()
        selected_size = st.selectbox('Size', size_options)

        graphics_options = df['Graphics_card'].unique()
        # Show options in Nepali and translate to 1 or 0
        graphics_dict = {'Yes ': 1, 'No ': 0}
        selected_graphics = st.selectbox('ग्राफिक्स कार्ड', list(graphics_dict.keys()))

    # Check if URL parameter exists
    url_params = st.experimental_get_query_params()
    if url_params:
        # Extract parameters from URL
        selected_brand = url_params['brand'][0]
        selected_model = url_params['model'][0]
        selected_processor = url_params['processor'][0]
        selected_ram = url_params['ram'][0]
        transformed_ssd = apply_log_transform(float(url_params['storage_ssd'][0]))
        selected_size = url_params['size'][0]
        selected_graphics = int(url_params['graphics_card'][0])

    # Predicting price based on user inputs or URL parameters
    if st.button('मूल्य पूर्वानुमान गर्नुहोस्'):
        # Prepare input data for prediction
        input_data = {
            'Brand': selected_brand,
            'Model': selected_model,
            'Processor': selected_processor,
            'RAM': selected_ram,
            'Storage SSD': transformed_ssd,
            'Size': selected_size,
            'Graphics_card': selected_graphics  # No need to translate here for API
        }

        # Get prediction from API
        predicted_price = get_prediction(input_data)

        # Display prediction result with enhanced styling
        st.subheader('मूल्य पूर्वानुमान')
        st.markdown(f'<div style="background-color:#f0f0f0;padding:10px;border-radius:10px;">'
                    f'<p style="font-size:24px;text-align:center;color:#008080;">'
                    f'तपाइँको चयन गरिएको ल्यापटपको मूल्य रु. <b>{predicted_price:.2f}</b>'
                    f'</p></div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
