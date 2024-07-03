import streamlit as st
import pandas as pd
import pickle
import numpy as np
import requests

# Load DataFrame from pickle file
@st.cache_data(hash_funcs={pd.DataFrame: lambda _: None})
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

# Main function to run Streamlit app
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

    # Predicting price based on user inputs
    if st.button('मूल्य पूर्वानुमान गर्नुहोस्'):
        # Prepare input data for prediction
        input_data = {
            'Brand': selected_brand,
            'Model': selected_model,
            'Processor': selected_processor,
            'RAM': selected_ram,
            'Storage SSD': transformed_ssd,
            'Size': selected_size,
            'Graphics_card': graphics_dict[selected_graphics]  # Translate back to 0 or 1
        }

        # Load the trained model from pickle file
        with open('random_forest_cv.pkl', 'rb') as f:
            model = pickle.load(f)

        # Make prediction
        X = pd.DataFrame([input_data])
        predicted_price = model.predict(X)[0]

        # Translate prediction to Nepali
        nepali_message = translate_to_nepali(input_data['Graphics_card'])

        # Display prediction result with enhanced styling
        st.subheader('मूल्य पूर्वानुमान')
        st.markdown(f'<div style="background-color:#f0f0f0;padding:10px;border-radius:10px;">'
                    f'<p style="font-size:24px;text-align:center;color:#008080;">'
                    f'तपाइँको चयन गरिएको ल्यापटपको मूल्य रु. <b>{predicted_price:.2f}</b>'
                    f'</p></div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
