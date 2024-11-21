import os
import tensorflow
import streamlit as st
import pickle as pk
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import base64


def add_bg_image(image_file):
    with open(image_file, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
    bg_image = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(bg_image, unsafe_allow_html=True)

# Add background image
add_bg_image("mandeep.png")

model = load_model(r"D:\NareshIT\Spyder_Homework\AI\ANN\ann_churn.h5")

st.title('Churn predictor')

st.write("The Customer Churn Predictor is a web application powered by machine learning, designed to forecast whether a customer is likely to leave or cancel their service. The application utilizes an Artificial Neural Network (ANN) model to analyze diverse customer data, such as demographic details, financial metrics, and activity history, providing an accurate prediction of churn likelihood.")

cities = ['Delhi','Banglore','Mumbai']
city = st.selectbox('Select city',cities)

genders = ['Male','Female']
gender = st.selectbox('Select Gender',genders)

age = st.number_input("Enter Age",min_value=18)

yesno = ['Yes','No']
creditcard = st.selectbox('Owns a creditcard',yesno)

creditscore = st.number_input('Enter Credit Score')

tenure = st.slider("Enter Tenure",min_value=0,max_value=15)

salary = st.number_input("Estimated salary")

balance = st.number_input("Enter Balance Amount")

number_products = st.slider("Enter Number of Product You Have",min_value=1,max_value=10)

activemember = st.selectbox('Active Account',yesno)

#Input Data
X_input = pd.DataFrame({
    'city_banglore':[0],
    'city_delhi':[0],
    'city_mumbai':[0],
    'CreditScore':[creditscore],
    'gender':[gender],
    'age':[age],
    'tenure':[tenure],
    'balance':[balance],
    'NumOfProducts':[number_products],
    'HasCreditCard':[creditcard],
    'ActiveMember':[activemember],
    'EstimatedSalary':[salary]
})


#One-hot encode the selected city
if city == 'Banglore':
    X_input['city_banglore'] = 1
elif city == 'Delhi':
    X_input['city_delhi'] = 1
elif city == 'Mumbai':
    X_input['city_mumbai'] = 1

#creditCard
if creditcard == 'Yes':
    X_input['HasCreditCard'] = 1
elif creditcard == 'No':
    X_input['HasCreditCard'] = 0

#activeMember
if activemember == 'Yes':
    X_input['ActiveMember'] = 1
elif activemember == 'No':
    X_input['ActiveMember'] = 0

#gender
if gender == 'Male':
    X_input['gender'] = 1
elif gender == 'Female':
    X_input['gender'] = 0

X_input_values = np.array([[
    X_input['city_banglore'][0],
    X_input['city_delhi'][0],
    X_input['city_mumbai'][0],
    X_input['CreditScore'][0],
    X_input['gender'][0],
    X_input['age'][0],
    X_input['tenure'][0],
    X_input['balance'][0],
    X_input['NumOfProducts'][0],
    X_input['HasCreditCard'][0],
    X_input['ActiveMember'][0],
    X_input['EstimatedSalary'][0]
]])

X_input_converted = pd.DataFrame(X_input_values, columns = X_input.columns)
sc=pk.load(open(r"D:\NareshIT\Spyder_Homework\AI\ANN\scalar.pkl","rb"))

Final_Input = sc.transform(X_input_converted)

if st.button("Predict"):
    prediction = model.predict(Final_Input)
    if prediction>=0.5:
        st.success("✅ Customer is not likely to churn. 🟢")
    else:
        st.error("Customer is likely to churn! ❌ ")

st.markdown("""
    ---
    <div style="text-align: center;">
        Created by Mandeepkaur😁<br>
        🚀 <a href="https://www.linkedin.com/in/mandeepkaur19/" target="_blank">LinkedIn</a>
    </div>
""", unsafe_allow_html=True)
