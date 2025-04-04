import streamlit as st
import pickle
import numpy as np

#import the model
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Laptop Price Prediction")

# brand
company = st.selectbox('Brand', df['Company'].unique())

# type of laptop
type = st.selectbox('Type', df['TypeName'].unique())

# ram
ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24])

# weight of laptop
weight = st.number_input('Weight of Laptop(in KG)', 0.0, 5.0, 1.0)

# touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

if touchscreen == 'Yes':
    touchscreen = 1
else:
    touchscreen = 0

# ips
ips = st.selectbox('IPS', ['No', 'Yes'])

if ips == 'Yes':
    ips = 1
else:
    ips = 0

# screen size
screen_size = st.number_input('Screen Size(inches)', 0.0, 20.0, 15.0)

# resolution
resolution = st.selectbox('Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '2560x1440', '1920x1200'])

# cpu
cpu = st.selectbox('CPU', df['Cpu Brand'].unique())

# gpu
gpu = st.selectbox('GPU', df['Gpu Brand'].unique())

# hdd
hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

# ssd
ssd = st.selectbox('SSD(in GB)', [0, 128, 256, 512, 1024])

# os
os = st.selectbox('OS', df['os'].unique())

# create a button for prediction
if st.button('Predict Price'):
    # calculating the PPI - Pixels Per Inch
    x_res = int(resolution.split('x')[0])
    y_res = int(resolution.split('x')[1])
    ppi = ((x_res ** 2) + (y_res ** 2)) ** 0.5 / screen_size
    # adding the ppi to the input data
    ppi = round(ppi, 2)

    # preprocess the input
    query = [[company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os]]
    prediction = np.exp(pipe.predict(query))
    st.title(f"Predicted Price: {prediction[0]:.2f} INR")


# Add a footer
st.markdown("""
    <div class="footer">
        <p>Developed by Suvam Naskar</p>
        <p>Contact: <a href="mailto:suvamnaskar.dev@gmail.com">suvamnaskar.dev@gmail.com</a></p>
        <p>GitHub: <a href="https://www.github.com/SuvamNaskar/">SuvamNaskar</a></p>
        <p>LinkedIn: <a href="https://www.linkedin.com/in/suvamnaskar/">suvamnaskar</a></p>
    </div>
""", unsafe_allow_html=True)