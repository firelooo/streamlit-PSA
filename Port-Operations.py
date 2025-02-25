import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sn
import pickle
import joblib
import streamlit.components.v1 as components
import numpy as np

st.set_page_config(page_title="PSA's Port Operations", page_icon=":ship:", layout="wide")

# -------- CSS STYLING ---------
st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
    }
    .header {
        font-size: 45px;
        color: #4B8BBE; /*COLOR*/
        font-weight: bold;
        text-align: center;
    }
    .subheader {
        font-size: 25px;
        color: black;
        font-weight: bold;
        text-align: center;
    }
    .text {
        font-size: 15px;
        color: black;
        font-weight: bold;
        text-align: center;
    }
    .invisible-button {
        background-color: transparent;
        color: black; 
        border-top: none;
        border-bottom: 1px solid #000000;
        border-left: none;  
        border-right: none; 
        width: 80%; 
        text-align: left;  
        font-size: 25px; 
        cursor: pointer;  /* Cursor change on hover */
        outline: none;  /* Remove outline on focus */
    }
    .invisible-button:hover {
        background-color: #e2e2e2;
        color: black;  
    }
    .title {
            font-size: 40px;
        font-weight: bold;
        color: #4B8BBE;
    }
    .prediction {
        font-size: 25px;
        color: #4B8BBE;
        font-weight: bold;
        text-align: left;
    }
    </style>
    """, unsafe_allow_html=True)

# Load your models (replace with actual model loading)
model1 = pickle.load(open('PortPeakHourPrediction.pkl', 'rb'))
model2 = pickle.load(open('PortWaitingTimePrediction.pkl', 'rb'))
model3 = pickle.load(open('PortEquipmentNeededPrediction.pkl', 'rb'))

if 'selected_page' not in st.session_state:
    st.session_state.selected_page = "Overview"

# Sidebar for navigation
st.sidebar.markdown('''<p style='text-align: left; font-size: 30px; font-weight: bold; color: #4B8BBE;'>Navigation</p>''', unsafe_allow_html=True)

# Navigation buttons
selected_page = st.sidebar.selectbox(
    "Explore:",
    options=["Overview", "Peak Times", "Idle Times", "Equipment Needs"]
)

st.sidebar.markdown(
    '''
    <div style="background-color: #ffffff; 
                border: 1px solid #ccc; 
                padding: 10px; 
                border-radius: 5px; 
                text-align: center; 
                margin-bottom: 10px;">
        <a href="http://127.0.0.1:5500/HTML/index.html" target="_blank" 
           style="color: black; 
                  text-decoration: none; 
                  font-size: 16px; 
                  display: block;">
            Main website
        </a>
    </div>
    ''',
    unsafe_allow_html=True
)

# Update the session state with the selected page
st.session_state.selected_page = selected_page

# Layout for Overview
if st.session_state.selected_page == "Overview":
    with st.container():
        st.markdown('''<p class="header">PSA's Port Operations</p>''', unsafe_allow_html=True)
        st.write("""<p class="text">PSA International (PSA) operates one of the world's largest
        and busiest port networks, with significant terminals in Singapore and beyond. This dashboard provides
        innovative data and AI-driven insights to optimize port resilience and efficiency across our global network.</p>""", unsafe_allow_html=True)

    st.markdown("-----------------------------------------------------------------------------------------------")

    with st.container():
        st.write('''<p class="subheader">Key performance indicators</p>''', unsafe_allow_html=True)

        # Add relevant KPIs or data visualizations
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""<div style='text-align: center; border: 2px solid #e6e6e6; border-radius: 5px; padding: 10px; background-color: #f9f9f9; margin-bottom: 20px;'>
                <h1>600</h1>
                <p>Ports connectivity</p>
                </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown("""<div style='text-align: center; border: 2px solid #e6e6e6; border-radius: 5px; padding: 10px; background-color: #f9f9f9; margin-bottom: 20px;'>
                <h1>100,000</h1>
                <p>TEUs handled daily</p>
                </div>""", unsafe_allow_html=True)

        # Create two columns for images
        col5, col6 = st.columns(2)

        with col5:
            st.image("port1.jpeg", use_container_width=True)
        
        with col6:
            st.image("container.jpeg", use_container_width=True)

        st.write("""<p style="text-align: left;">Our AI-driven insights and data analytics aims to provide real-time information, head to our main website for
        more information.</p>""", unsafe_allow_html=True) 

        # Add text below the images
        html_link = '''
            <a href="http://127.0.0.1:5500/HTML/index.html" target="_blank" /*Live server link*/
            style="background-color: #f0f0f0; /* Your desired background color */
                color: black; 
                text-decoration: none; 
                display: inline-block; 
                padding: 10px; 
                border-radius: 5px; 
                text-align: center; 
                cursor: pointer;">
                Main website
            </a>'''
        components.html(html_link, height=50)

# Layout for Model 1 Analysis
elif st.session_state.selected_page == "Peak Times":
    with st.container():
        st.markdown('''<p class="header">Peak Times Prediction</p>''', unsafe_allow_html=True)
        st.write("""<p class="text">Our predictive model aims to forecast the peak hours at the port, providing valuable insights to optimize
                 resource allocation, improve traffic management, and enhance operational efficiency by anticipating high-demand periods.
        </p>""", unsafe_allow_html=True)
    
    with st.container():
        st.header("Analysis")
        st.image("PSA-worldwide-map.png", caption='Worldwide Map', use_container_width=True)

    with st.container():
        def user_input_features():
        # Create 2x2 grid for input options
            col1, col2 = st.columns(2)

            # First column inputs
            with col1:
                # Dropdown for Weather Conditions
                weather_conditions = ['Clear', 'Rainy', 'Cloudy']
                selected_weather = st.selectbox('Select Weather Conditions:', weather_conditions, key='weather')

                # Slider for Number of Ships
                selected_number_of_ships = st.slider('Number of Ships:', min_value=0, max_value=10, value=1, key='number_of_ships')

                # Slider for Cargo Volume
                selected_cargo_volume = st.slider('Cargo Volume (tons):', min_value=0.0, max_value=100.0, value=1.0, step=0.1, key='cargo_volume')

            # Second column inputs
            with col2:
                # Dropdown for Berth Assigned
                berth_assigned = ['B1', 'B2', 'B3', 'B4', 'B5']
                selected_berth = st.selectbox('Select Berth Assigned:', berth_assigned, key='berth')

                selected_holiday_period = st.selectbox('Holiday Period:', ['No', 'Yes'], key='holiday_period')

                # Convert selected value to binary for processing if needed
                is_holiday_period = 1 if selected_holiday_period == 'Yes' else 0

                # Day Type input
                day_type = ['Weekday', 'Weekend']
                selected_day_type = st.selectbox('Select Day Type:', day_type, key='day_type')

                # Arrival Hour slider
            selected_arrival_hour = st.slider('Arrival Hour (24-hour format):', min_value=0, max_value=23, value=12, key='arrival_hour')
            st.write(f"Selected Arrival Hour: {selected_arrival_hour:02d}:00")

            # Creating a dictionary to hold the user inputs
            data = {
                'Cargo_Volume (tons)': selected_cargo_volume,
                'Berth_Assigned': selected_berth,
                'Number_of_Ships': selected_number_of_ships,
                'Weather_Conditions': selected_weather,
                'Holiday_Period': is_holiday_period,
                'Day_Type': selected_day_type,
                'Arrival_Hour': selected_arrival_hour,
            }

            # Convert to DataFrame
            features = pd.DataFrame(data, index=[0])

            return features

        user_df = user_input_features()

        if st.button("Submit"):
            # Create a dictionary to store user inputs
            user_inputs = user_df.to_dict(orient='records')[0]

            formatted_arrival_hour = f"{user_inputs['Arrival_Hour']:02d}:00"
    
            user_inputs['Arrival_Hour'] = formatted_arrival_hour

            # Convert the dictionary to a DataFrame
            user_inputs_df = pd.DataFrame([user_inputs])
            
            st.subheader("User Inputs")
            st.table(user_inputs_df)

            st.subheader("Results")
            prediction = model1.predict(user_df)
            
            prediction_result = "Yes" if prediction[0] == 1 else "No"

            st.markdown(f'<p class="prediction">Predicted Peak Hour:  {prediction_result}!</p>', unsafe_allow_html=True)

# Layout for Model 2 Analysis
elif st.session_state.selected_page == "Idle Times":
    with st.container():
        st.markdown('''<p class="header">Idle Times Prediction</p>''', unsafe_allow_html=True)
        st.write("""<p class="text">Our predictive model aims to forecast the idle times of ships at the port, providing valuable insights to optimize
        resource allocation and operational efficiency. The model takes into account various factors such as weather conditions, time of day, and cargo volume.</p>""", unsafe_allow_html=True)

# Image Section
    with st.container():
        col1, col2, = st.columns(2)

        with col1:
            st.image("weather-forecast.jpg", caption='live weather forecast', use_container_width=True)

        with col2:
            st.image("calandar.jpg", caption='Singapore Calandar 2024', use_container_width=True)

        st.markdown("""<hr style="height:2px;border:none;color:#4B8BBE;background-color:black;" />""", unsafe_allow_html=True)

    # User Input Features
    def user_input_features():
        # Create 2x2 grid for input options
        col1, col2 = st.columns(2)

        # First column inputs
        with col1:
            selected_holiday_period = st.selectbox('Holiday Period:', ['No', 'Yes'])
            is_holiday_period = 1 if selected_holiday_period == 'Yes' else 0

            weather_conditions = ['Clear', 'Rainy', 'Cloudy']
            selected_weather = st.selectbox('Select Weather Conditions:', weather_conditions)

            time_of_day = ['Morning', 'Afternoon', 'Evening', 'Night']
            selected_time_of_day = st.selectbox('Select Time of Day:', time_of_day)

            selected_number_of_ships = st.slider('Number of Ships:', min_value=0, max_value=10, value=1)

            selected_cargo_volume = st.slider('Cargo Volume (tons):', min_value=0.0, max_value=100.0, value=1.0, step=0.1)
            
        # Second column inputs
        with col2:
            selected_peak_hour = st.selectbox('Peak Hour?:', ['No', 'Yes'])
            is_peak_hour = 1 if selected_peak_hour == 'Yes' else 0

            selected_day_type = st.selectbox('Select Day Type:', ['Weekday', 'Weekend'])

            berth_assigned = ['B1', 'B2', 'B3', 'B4', 'B5']
            selected_berth = st.selectbox('Select Berth Assigned:', berth_assigned, key='berth')

            selected_vessel_size = st.slider('Vessel Size (meters):', min_value=0, max_value=300, value=50)

            selected_arrival_hour = st.slider('Arrival Hour (24-hour format):', min_value=0, max_value=23, value=12)
            st.write(f"Selected Arrival Hour: {selected_arrival_hour:02d}:00")

        # Creating a dictionary to hold the user inputs
        data = {
            'Cargo_Volume (tons)': selected_cargo_volume,
            'Vessel_Size (meters)': selected_vessel_size,
            'Number_of_Ships': selected_number_of_ships,
            'Peak_Hour': is_peak_hour,
            'Holiday_Period': is_holiday_period,
            'Arrival_Hour': selected_arrival_hour,
            'Berth_Assigned_B2': 0,
            'Berth_Assigned_B3': 0,
            'Berth_Assigned_B4': 0,
            'Berth_Assigned_B5': 0,
            'Weather_Conditions_Cloudy': 1 if selected_weather == 'Cloudy' else 0,
            'Weather_Conditions_Rainy': 1 if selected_weather == 'Rainy' else 0,
            'Time_of_Day_Evening': 0,
            'Time_of_Day_Morning': 0,
            'Time_of_Day_Night': 0,
            'Day_Type_Weekend': 1 if selected_day_type == 'Weekend' else 0,  # Binary encoding for Day_Type
        }

        if selected_berth == 'B2':
            data['Berth_Assigned_B2'] = 1
        elif selected_berth == 'B3':
            data['Berth_Assigned_B3'] = 1
        elif selected_berth == 'B4':
            data['Berth_Assigned_B4'] = 1
        elif selected_berth == 'B5':
            data['Berth_Assigned_B5'] = 1

        if selected_time_of_day == 'Morning':
            data['Time_of_Day_Morning'] = 1
        elif selected_time_of_day == 'Evening':
            data['Time_of_Day_Evening'] = 1
        elif selected_time_of_day == 'Night':
            data['Time_of_Day_Night'] = 1
        

        # Convert to DataFrame
        features = pd.DataFrame(data, index=[0])

        return features

    user_df = user_input_features()

    # Submit Button
    if st.button("Submit"):
        # Create a dictionary to store user inputs
        user_inputs = user_df.to_dict(orient='records')[0]

        user_inputs_df = pd.DataFrame([user_inputs])
        
        st.subheader("User Inputs")
        st.table(user_inputs_df)

        st.subheader("Results")
        prediction_result = model2.predict(user_df)
        prediction_result = np.around(prediction_result[0], 2) # Assuming model1 is defined elsewhere
        st.markdown(f'<p class="prediction">Predicted Idle time:  {prediction_result} Hours!</p>', unsafe_allow_html=True)

# Layout for Model 3 Analysis
elif st.session_state.selected_page == "Equipment Needs":
    with st.container():
        st.markdown('''<p class="header">Idle Times Prediction</p>''', unsafe_allow_html=True)
        st.write("""<p class="text">Our model aims to predict the necessary number of equipment required for smooth port operations. 
                 The primary goal is to optimize the deployment of equipment such as cranes, forklifts, and loaders, ensuring that they are 
                 used efficiently to minimize delays and increase productivity.
        </p>""", unsafe_allow_html=True)

    st.markdown("""<hr style="height:2px;border:none;color:#4B8BBE;background-color:black;" />""", unsafe_allow_html=True)

    st.subheader("Equipment Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""<div style='text-align: center; border: 2px solid #e6e6e6; border-radius: 5px; padding: 10px; background-color: #f9f9f9; margin-bottom: 20px;'>
                <h4>Container Cranes</h4>
                <p>150</p>
                </div>""", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""<div style='text-align: center; border: 2px solid #e6e6e6; border-radius: 5px; padding: 10px; background-color: #f9f9f9; margin-bottom: 20px;'>
                <h4>Container Trucks</h4>
                <p>1,000</p>
                </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown("""<div style='text-align: center; border: 2px solid #e6e6e6; border-radius: 2px; padding: 10px; background-color: #f9f9f9; margin-bottom: 20px;'>
                <h4>Forklifts</h4>
                <p>3,000</p>
                </div>""", unsafe_allow_html=True)

    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown("""<div style='text-align: center; border: 2px solid #e6e6e6; border-radius: 2px; padding: 10px; background-color: #f9f9f9; margin-bottom: 20px;'>
                <h4>Straddle Carriers</h4>
                <p>10,000</p>
                </div>""", unsafe_allow_html=True)

    with col5:
        st.markdown("""<div style='text-align: center; border: 2px solid #e6e6e6; border-radius: 2px; padding: 10px; background-color: #f9f9f9; margin-bottom: 20px;'>
                <h4>Automated Vehicles</h4>
                <p>500</p>
                </div>""", unsafe_allow_html=True)

    with col6:
        st.markdown("""<div style='text-align: center; border: 2px solid #e6e6e6; border-radius: 2px; padding: 10px; background-color: #f9f9f9; margin-bottom: 20px;'>
                    <h4>Containers</h4>
                    <p>100,000</p>
                    </div>""", unsafe_allow_html=True)

    with st.container():
        def user_input_features():
            # Create 2x2 grid for input options
            col1, col2 = st.columns(2)

            # First column inputs
            with col1:

                weather_conditions = ['Clear', 'Rainy', 'Cloudy']
                selected_weather = st.selectbox('Select Weather Conditions:', weather_conditions)

                time_of_day = ['Morning', 'Afternoon', 'Evening', 'Night']
                selected_time_of_day = st.selectbox('Select Time of Day:', time_of_day)

                selected_waiting_time = st.slider('Waiting Time (hours):', min_value=0.0, max_value=24.0, value=1.0, step=0.1)

                selected_cargo_volume = st.slider('Cargo Volume (tons):', min_value=0.0, max_value=100.0, value=0.1)
                
            # Second column inputs
            with col2:
                selected_peak_hour = st.selectbox('Peak Hour?:', ['No', 'Yes'])
                is_peak_hour = 1 if selected_peak_hour == 'Yes' else 0

                selected_day_type = st.selectbox('Select Day Type:', ['Weekday', 'Weekend'])

                berth_assigned = ['B1', 'B2', 'B3', 'B4', 'B5']
                selected_berth = st.selectbox('Select Berth Assigned:', berth_assigned, key='berth')

                selected_vessel_size = st.slider('Vessel Size (meters):', min_value=0.0, max_value=500.0, value=0.1)

            # Creating a dictionary to hold the user inputs
            data = {
                'Cargo_Volume (tons)': selected_cargo_volume,
                'Vessel_Size (meters)': selected_vessel_size,
                'Waiting_Time (hours)': selected_waiting_time,
                'Peak_Hour': is_peak_hour,
                'Berth_Assigned_B2': 0,
                'Berth_Assigned_B3': 0,
                'Berth_Assigned_B4': 0,
                'Berth_Assigned_B5': 0,
                'Weather_Conditions_Cloudy': 1 if selected_weather == 'Cloudy' else 0,
                'Weather_Conditions_Rainy': 1 if selected_weather == 'Rainy' else 0,
                'Time_of_Day_Evening': 0,
                'Time_of_Day_Morning': 0,
                'Time_of_Day_Night': 0,
                'Day_Type_Weekend': 1 if selected_day_type == 'Weekend' else 0,  # Binary encoding for Day_Type
            }

            if selected_berth == 'B2':
                data['Berth_Assigned_B2'] = 1
            elif selected_berth == 'B3':
                data['Berth_Assigned_B3'] = 1
            elif selected_berth == 'B4':
                data['Berth_Assigned_B4'] = 1
            elif selected_berth == 'B5':
                data['Berth_Assigned_B5'] = 1

            if selected_time_of_day == 'Morning':
                data['Time_of_Day_Morning'] = 1
            elif selected_time_of_day == 'Evening':
                data['Time_of_Day_Evening'] = 1
            elif selected_time_of_day == 'Night':
                data['Time_of_Day_Night'] = 1
            

            # Convert to DataFrame
            features = pd.DataFrame(data, index=[0])

            return features

        user_df = user_input_features()

        # Submit Button
        if st.button("Submit"):
            # Create a dictionary to store user inputs
            user_inputs = user_df.to_dict(orient='records')[0]

            user_inputs_df = pd.DataFrame([user_inputs])
            
            st.subheader("User Inputs")
            st.table(user_inputs_df)

            st.subheader("Results")
            prediction_result = model3.predict(user_df)
            prediction_result = np.around(prediction_result[0], 0) # Assuming model1 is defined elsewhere
            st.markdown(f'<p class="prediction">Predicted Equipment needed:  {prediction_result}!</p>', unsafe_allow_html=True)

# -------- FOOTER SECTION ---------
with st.container():
    st.markdown("""<hr style="height:2px;border:none;color:#4B8BBE;background-color:black;" />""", unsafe_allow_html=True)
    st.write("""
        Contact us: MarkSparkle@gmail.com |    Follow us on [Twitter](https://twitter.com)    [Facebook](https://facebook.com)
        """)

