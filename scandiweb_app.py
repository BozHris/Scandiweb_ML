import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import os


print(os.listdir())




try:
    classification_model = tf.keras.models.load_model(r'C:\Users\Bozo\Desktop\Scandiweb_ML\Classification_model.h5')
    regression_model = load_model(r'C:\Users\Bozo\Desktop\Scandiweb_ML\Regression_model.h5')
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


item_categories = [
    '(not set)', 'Amazon Devices', 'Appliances', 'Arts, Crafts & Sewing',
    'Automotive', 'Automotive Parts & Accessories', 'Baby', 'Baby Products',
    'Beauty & Personal Care', 'Books', 'CDs & Vinyl', 'Cell Phones & Accessories',
    'Clothing, Shoes & Jewelry', 'Collectibles & Fine Art', 'Electronics',
    'Entertainment Collectibles', 'Garden & Outdoor', 'Grocery & Gourmet Food',
    'Handmade', 'Handmade Products', 'Health & Household', 'Health, Household & Baby Care',
    'Home & Kitchen', 'Industrial & Scientific', 'Lighting Assemblies & Accessories',
    'Magazine Subscriptions', 'Movies & TV', 'Musical Instruments', 'Office Products',
    'Other', 'Patio, Lawn & Garden', 'Pet Supplies', 'Sports & Outdoors',
    'Todos los departamentos', 'Tools & Home Improvement', 'Toys & Games', 'Video Games',
    'Wearable Technology'
]

channels = [
    'Organic Social', 'Unassigned', 'Paid Social', 'Direct', 'Display', 'Paid Search',
    'Mobile Push Notifications', 'Organic Search', 'Email', 'Referral', 'Organic Shopping'
]

devices = ['mobile', 'desktop', 'smart tv', 'tablet']

# Streamlit app
st.title("Shopping Prediction App")

# Inputs
item_category = st.selectbox("Select Item Category:", item_categories)
channel = st.selectbox("Select Channel:", channels)
device = st.selectbox("Select Device:", devices)

# Single number input for visits
visits_count = st.number_input("How many times did the user visit the website?", min_value=0, step=1)

# Set device count same as visits count
device_count = visits_count

if st.button('Predict'):
    # Prepare the input data
    data = {
        '(not set)': [0], 'Amazon Devices': [0], 'Appliances': [0], 'Arts, Crafts & Sewing': [0],
        'Automotive': [0], 'Automotive Parts & Accessories': [0], 'Baby': [0], 'Baby Products': [0],
        'Beauty & Personal Care': [0], 'Books': [0], 'CDs & Vinyl': [0], 'Cell Phones & Accessories': [0],
        'Clothing, Shoes & Jewelry': [0], 'Collectibles & Fine Art': [0], 'Electronics': [0],
        'Entertainment Collectibles': [0], 'Garden & Outdoor': [0], 'Grocery & Gourmet Food': [0],
        'Handmade': [0], 'Handmade Products': [0], 'Health & Household': [0], 'Health, Household & Baby Care': [0],
        'Home & Kitchen': [0], 'Industrial & Scientific': [0], 'Lighting Assemblies & Accessories': [0],
        'Magazine Subscriptions': [0], 'Movies & TV': [0], 'Musical Instruments': [0], 'Office Products': [0],
        'Other': [0], 'Patio, Lawn & Garden': [0], 'Pet Supplies': [0], 'Sports & Outdoors': [0],
        'Todos los departamentos': [0], 'Tools & Home Improvement': [0], 'Toys & Games': [0], 'Video Games': [0],
        'Wearable Technology': [0], 'Organic Social': [0], 'Unassigned': [0], 'Paid Social': [0],
        'Direct': [0], 'Display': [0], 'Paid Search': [0], 'Mobile Push Notifications': [0],
        'Organic Search': [0], 'Email': [0], 'Referral': [0], 'Organic Shopping': [0],
        'mobile': [0], 'desktop': [0], 'smart tv': [0], 'tablet': [0]
    }

    # Update data based on user inputs
    data[item_category][0] = visits_count
    data[channel][0] = visits_count
    data[device][0] = device_count

    # Convert to DataFrame
    input_df = pd.DataFrame(data)

    st.write("Input Data:", input_df)

    # Make classification prediction
    classification_prediction = classification_model.predict(input_df)

    if classification_prediction[0][0] > 0.6:
        st.write("There is a probability of a purchase!")
        # Make regression prediction
        regression_prediction = regression_model.predict(input_df)
        if regression_prediction.ndim == 2 and regression_prediction.shape[1] == 1:

            final_result = round(regression_prediction[0][0], 2)
        elif regression_prediction.ndim == 1:

            final_result = round(regression_prediction[0], 2)
        else:

            final_result = round(regression_prediction, 2)
        st.write(f"Estimated Total Revenue: {final_result}$")
    else:
        st.write("Low probability of purchase.")
