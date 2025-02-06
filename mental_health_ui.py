import streamlit as st
import pickle
import joblib
import pandas as pd
from io import BytesIO
from gtts import gTTS

st.set_page_config(
    page_title="Mental_Health_Prediction",
    page_icon="mental_health_icon.png",
    layout="wide")



st.title("Mental Health Prediction")

#Loading the feature values
ct_loaded = joblib.load('feature_values')

#Loading the model
file_name = "mental_health_prediction_model"
model = pickle.load(open(file_name,"rb"))

name = st.text_input("Enter your name:")

col1, col2, col3 = st.columns(3)

age = col1.number_input("Enter your Age:", min_value=0, max_value=100, value=18)

gender = col2.selectbox(
    "Gender:",
    ["Male", "Female", "Non-binary"]
)

self_employed = col3.selectbox(
    "Are you self employed?",
    ["Yes", "No"]
)

famiy_history = col1.selectbox(
    "Do you have a family history of mental illness",
    ["Yes", "No"]
)

work_interfere = col2.selectbox(
    "If you have a mental health condition, do you feel that it interferes with your work?",
    ["Never", "Rarely", "Sometimes", "Often", "N/A"]
)

no_employees = col3.selectbox(
    "How many employees does your company or organization have?",
    ["1-5", "6-25","26-100", "100-500", "500-1000", "More than 1000"]
)

remote_work = col1.selectbox(
    "Do you work remotely (outside of an office) atleast 50% of the time?",
    ["Yes", "No"]
)

tech_company = col2.selectbox(
    "Is your employer primarily a tech company organization?",
    ["Yes", "No"]
)

benefits = col3.selectbox(
    "Does your employer provide mental health benefits?",
    ["Yes", "No", "Don't know"]
)

care_options = col1.selectbox(
    "Do you know the option for health care your employer provides?",
    ["Yes", "No", "Not sure"]
)

wellness_program = col2.selectbox(
    "Have your employer ever discussed mental health as part of an employee wellness program?",
    ["Yes", "No", "Don't know"]
)

seek_help = col3.selectbox(
    "Do your employer provide resources to learn more about mental health issues and how to seek help?",
    ["No", "Yes", "Don't know"]
)

anonymity = col1.selectbox(
    "Is your anonymity protected if you choose to take advantage of mental health issues and how to seek help?",
    ["Don't know", "Yes", "No"]
)

leave = col2.selectbox(
    "How easy is it for you to take medical leave for a mental health condition?",
    ["Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult", "Don't know"]
)

mental_health_consequence = col3.selectbox(
    "Do  you think that discussing a mental health issue with your employer would have negative consequences?",
    ["No", "Yes", "Maybe"]
)

phys_health_consequence = col1.selectbox(
    "Do  you think that discussing a physical health issue with your employer would have negative consequences?",
    ["No", "Yes", "Maybe"]
)

coworkers = col2.selectbox(
    "Would you be willing to discuss a mental health issue with your coworkers?",
    ["Some of them", "Yes", "No"]
)

supervisor = col3.selectbox(
    "Would you be willing to discuss a mental health issue with your direct supervisor(s)?",
    ["Yes", "No", "Some of them"]
)

mental_health_interview = col1.selectbox(
    "Would you bring up a mental health issue with a potential employer in an interview?",
    ["No", "Yes", "Maybe"]
)

phys_health_interview = col2.selectbox(
    "Would you bring up a physical health issue with a potential employer in an interview?",
    ["Maybe", "Yes", "No"]
)

mental_vs_physical = col3.selectbox(
    "Do you feel that your employer takes mental health as seriously as physical health?",
    ["Don't know", "Yes", "No"]
)

obs_consequence = col1.selectbox(
    "Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?",
    ["Yes", "No"]
)


if st.button("Predict"):
    age_1 = []
    gender_2 = []
    self_employed_3 = []
    family_history_4 = []
    work_interfere_5 = []
    no_employees_6 = []
    remote_work_7 = []
    tech_company_8 = []
    benefits_9 = []
    care_options_10 = []
    wellness_program_11 = []
    seek_help_12 = []
    anonymity_13 = []
    leave_14 = []
    mental_health_consequence_15 = []
    phys_health_consequence_16 =[]
    coworkers_17 = []
    supervisor_18 = []
    mental_health_interview_19 = []
    phys_health_interview_20 = []
    mental_vs_physical_21 = []
    obs_consequence_22 = []

    age_1.append(age), gender_2.append(gender), self_employed_3.append(self_employed), family_history_4.append(famiy_history),
    work_interfere_5.append(work_interfere), no_employees_6.append(no_employees), remote_work_7.append(remote_work),
    tech_company_8.append(tech_company), benefits_9.append(benefits), care_options_10.append(care_options),
    wellness_program_11.append(wellness_program), seek_help_12.append(seek_help), anonymity_13.append(anonymity), leave_14.append(leave),
    mental_health_consequence_15.append(mental_health_consequence), phys_health_consequence_16.append(phys_health_consequence),
    coworkers_17.append(coworkers), supervisor_18.append(supervisor), mental_health_interview_19.append(mental_health_interview),
    phys_health_interview_20.append(phys_health_interview), mental_vs_physical_21.append(mental_vs_physical),
    obs_consequence_22.append(obs_consequence)

    #Creating dataframe
    data = {
        'Age': age_1, 'Gender': gender_2, 'self_employed': self_employed_3, 'family_history': family_history_4,
        'work_interfere': work_interfere_5, 'no_employees': no_employees_6, 'remote_work': remote_work_7,
        'tech_company': tech_company_8, 'benefits': benefits_9, 'care_options': care_options_10, 'wellness_program': wellness_program_11,
        'seek_help': seek_help_12, 'anonymity': anonymity_13, 'leave': leave_14, 'mental_health_consequence': mental_health_consequence_15,
        'phys_health_consequence': phys_health_consequence_16, 'coworkers': coworkers_17, 'supervisor': supervisor_18,
        'mental_health_interview': mental_health_interview_19, 'phys_health_interview': phys_health_interview_20,
        'mental_vs_physical': mental_vs_physical_21, 'obs_consequence': obs_consequence_22
    }

    x = pd.DataFrame(data)

    #Encoding the features
    x = ct_loaded.transform(x)

    #Predicting
    treatment = model.predict(x)

    text = ''
    if treatment == 0:
        text = f"{name}, you do not need any type of Mental Health Treatment."
        st.success(text)

    elif treatment == 1:
        text = f"{name}, you need Mental Health Treatment."
        st.error(text)

    sound_file = BytesIO()
    tts = gTTS(text, lang='en')
    tts.write_to_fp(sound_file)

    st.audio(sound_file, format='audio/ogg', autoplay=True)

