import streamlit as st
import requests

st.title("ML Prediction App")

# Input for features as per the required format
st.write("Enter your features:")

# Collect the input as specified
age = st.number_input("Age", value=0.0)
gender = st.number_input("Gender if male 1 else 0", value=0)  # Assuming gender is categorical
education = st.number_input("Education Level", value=0)  # Replace with appropriate input method
introversion_score = st.number_input("Introversion Score", value=0.0)
sensing_score = st.number_input("Sensing Score", value=0.0)
thinking_score = st.number_input("Thinking Score", value=0.0)
judging_score = st.number_input("Judging Score", value=0.0)
interest = st.number_input("Interest 0 if unknown, 1 if arts, 2 if Others, 3 if Technology, 4 if Sport", value=0)
# interest = st.selectbox("Interest Score", ("Unknown",'Arts','Others','Technology','Sports'))

# Create a list from gathered inputs
features = [
    age,
    gender, 
    education,
    introversion_score,
    sensing_score,
    thinking_score,
    judging_score,
    interest
]

if st.button("Predict"):
    # Prepare input data for request

    input_data = {
           "Age": age,
           "Gender": gender,
           "Education": education,
           "Introversion_score": introversion_score,
           "Sensing_score": sensing_score,
           "Thinking_score": thinking_score,
           "Judjing_score": judging_score,
           "Interest": interest
       }
    print(input_data)
    # Send request to FastAPI service
    response = requests.post("http://fastapi:8000/predict", json=input_data)
    
    if response.status_code == 200:
        prediction = response.json().get("prediction")
        st.write(f"To check the answer see the map:\n 'ENFP': 0,'ESFP': 1,'INTP': 2,'INFP': 3,'ENFJ': 4,'ENTP': 5,'ESTP': 6, 'ISTP': 7, 'INTJ': 8, 'INFJ': 9, 'ISFP': 10, 'ENTJ': 11, 'ESFJ': 12,'ISFJ': 13,'ISTJ': 14,'ESTJ': 15 The predicted value is: {prediction}")
    else:
        st.error("Error in prediction request.")
