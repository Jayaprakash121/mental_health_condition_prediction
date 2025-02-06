import pickle
import joblib
import pandas as pd

#Loading the feature values
ct_loaded = joblib.load('feature_values')

#Loading the model
file_name = "mental_health_prediction_model"
model = pickle.load(open(file_name,"rb"))

#Giving custom inputs, change the below according to the symptoms
age = 20
gender = ''
self_employed = ''
famiy_history = ''
work_interfere = ''
no_employees = ''
remote_work = ''
tech_company = ''
benefits = ''
care_options = ''
wellness_program = ''
seek_help = ''
anonymity = ''
leave = ''
mental_health_consequence = ''
phys_health_consequence = ''
coworkers = ''
supervisor = ''
mental_health_interview = ''
phys_health_interview = ''
mental_vs_physical = ''
obs_consequence = ''

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

if treatment == 0:
    print("NO")
elif treatment == 1:
    print("YES")