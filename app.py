import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix 

s = pd.read_csv("social_media_usage.csv",
                na_values="UNKNOWN")

#turns a variable into a binary
def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x

#turns gender into a binary variable
def gender_bender(x):
    x = np.where(x == 2, 1, 0)
    return x

#remove missing data and creating a new subset of the data
ss = s[["income", "educ2", "par", "marital", "gender", "age", "web1h"]]

#creat and clean Linkdin user colum
ss = ss[ss["web1h"] <= 2]
ss["web1h"] = ss["web1h"].apply(clean_sm)
ss.rename(columns = {"web1h" : "sm_li"}, inplace = True)

#make gender binary and rename as female
ss = ss[ss["gender"] <= 2] 
ss["gender"] = ss["gender"].apply(gender_bender)
ss.rename(columns = {"gender" : "female"}, inplace = True)

#make parent binary
ss = ss[ss["par"] <= 2]
ss["par"] = ss["par"].apply(clean_sm)

#make marital binary
ss = ss[ss["marital"] <= 6]
ss["marital"] = ss["marital"].replace(4,1)
ss["marital"] = ss["marital"].replace([2, 3, 5, 6],0)

#removing missing values for age, education, and income
ss = ss[ss["age"] <= 97]
ss = ss[ss["educ2"] <= 8]
ss = ss[ss["income"] <= 9]

#create the x and y vectors
x_ss = ss.drop("sm_li", axis = 1)
y_ss = ss["sm_li"]

#create our test and train data sets
X_train, X_test, y_train, y_test = train_test_split(x_ss,
                                                    y_ss,
                                                    stratify = y_ss,
                                                    test_size = 0.2,
                                                    random_state = 987)

#create and fit our logistic regression model
lr = LogisticRegression(class_weight = "balanced")

lr.fit(X_train, y_train)

st.markdown("# Welcome to the Linkdin User prediction app!")
st.markdown("### Please enter the information for a person below and we'll predict if they are a Linkdin user.")

#user input for income level
income = st.selectbox("Income level", 
              options = ["Less than $10,000",
                         "10 to under $20,000",
                         "20 to under $30,000",
                         "30 to under $40,000",
                         "40 to under $50,000",
                         "50 to under $75,000",
                         "75 to under $100,000",
                         "100 to under $150,000",
                         "$150,000 or more"])

if income == "Less than $10,000":
     income = 1
elif income == "10 to under $20,000":
     income = 2
elif income == "20 to under $30,000":
    income = 3
elif income == "30 to under $40,000":
     income = 4
elif income == "40 to under $50,000":
    income = 5
elif income == "50 to under $75,000":
     income = 6
elif income == "75 to under $100,000":
    income = 7
elif income == "100 to under $150,000":
    income = 8
else:
     income = 9

#user input for education level
educ = st.selectbox("Education level", 
              options = ["Less than high school (Grades 1-8 or no formal schooling)",
                         "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)",
                         "High school graduate (Grade 12 with diploma or GED certificate)",
                         "Some college, no degree (includes some community college)",
                         "Two-year associate degree from a college or university",
                         "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)",
                         "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)",
                         "Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)"])

if educ == "Less than high school (Grades 1-8 or no formal schooling)":
     educ = 1
elif educ == "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)":
     educ = 2
elif educ == "High school graduate (Grade 12 with diploma or GED certificate)":
    educ = 3
elif educ == "Some college, no degree (includes some community college)":
     educ = 4
elif educ == "Two-year associate degree from a college or university":
    educ = 5
elif educ == "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)":
     educ = 6
elif educ == "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)":
    educ = 7
else:
     educ = 8
    
#user input for parent status
par = st.radio("Parental status",
               ["Yes",
                "No"])

if par == "Yes":
     par = 1
else:
     par = 0

#user input for marital status
marital = st.radio("Marital status",
               ["Yes",
                "No"])

if marital == "Yes":
     marital = 1
else:
     marital = 0

#user input for gender
female = st.radio("Gender",
               ["Female",
                "Male"])

if female == "Yes":
     female = 1
else:
     female = 0     

#user input for age
age = st.slider(label="Enter Age",
           min_value=18,
           max_value=98,
           value=50)

#predicts if our input variables will result in a Linkdin user
person1 = [income, educ, par, marital, female, age]

predicted_class = lr.predict([person1])
probs = lr.predict_proba([person1])

if predicted_class == 1:
     pc = "Linkdin user"
else:
     pc = "not a Linkdin user"

probability = (round(probs[0][1],2)*100)


st.write(f"We predict that this person is {pc}.") 
st.write(f"Probability that this person is Linkdin user: {probability}%")