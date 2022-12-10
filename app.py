import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

def user_input_features():
    st.write("""**1. Select Age :**""")
    age = st.slider('', 0, 100, 25)
    st.write("""**You selected this option **""",age)

    st.write("""**2. Select Gender :**""")
    sex = st.selectbox("(1 = Male, 0 = Female)",[1,0])
    st.write("""**You selected this option **""",sex)

    st.write("""**3. Select Chest Pain Type :**""")
    cp = st.selectbox("(0 = Typical Angina, 1 = Atypical Angina, 2 = Non-Anginal Pain, 3 = Asymptotic)",[0,1,2,3])
    st.write("""**You selected this option **""",cp)

    st.write("""**4. Select Resting Blood Pressure :**""")
    rbp = st.slider('In mm/Hg unit', 0, 200, 110)
    st.write("""**You selected this option **""",rbp)

    st.write("""**5. Select Select Serum Colestrol :**""")
    ssc = st.slider('In mg/dl unit', 0, 600, 110)
    st.write("""**You selected this option **""",ssc)

    st.write("""**6. Maximum Heart Rate Achieved (THALACH) :**""")
    hra = st.slider('', 0, 220, 110)
    st.write("""**You selected this option **""",hra)

    st.write("""**7. Exercise Induced Angina (Pain in chest while exercise) :**""")
    ea = st.selectbox("(1 = Yes, 0 = No)",[0,1])
    st.write("""**You selected this option **""",ea)

    st.write("""**8. Oldpeak (ST depression induced by exercise relative to rest)) :**""")
    op = st.slider('', 0.00, 10.00, 2.00)
    st.write("""**You selected this option **""",op)

    st.write("""**9. Slope (The slope of the peak exercise ST segment) :**""")
    sl = st.selectbox("(0 = upsloping , 1 = flat , 2 = downsloping)",[0,1,2])
    st.write("""**You selected this option **""",sl)

    st.write("""**10. CA (Number of major vessels (0-3) coloured by flouroscopy) :**""")
    ca = st.selectbox("(Select 0,1,2 or 3)",[0,1,2,3])
    st.write("""**You selected this option **""",ca)

    st.write("""**11. Thal (Thalassemia) :**""")
    tl = st.slider('1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect', 0, 3, 2)
    st.write("""**You selected this option **""",tl)

    return [[age,sex,cp,rbp,ssc,hra,ea,op,sl,ca,tl]]

def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://free4kwallpapers.com/uploads/originals/2020/12/17/heartbleed-wallpaper.jpg");
            background-attachment: fixed;
            background-size : cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
     )

add_bg_from_url() 

st.title('Heart Disease Prediction using Machine Learning')
st.header('- A S JAGANATH THILAK')
st.markdown('---')
st.subheader('Fill in the below form to predict heart disease')
st.markdown('---')
df = user_input_features()
# st.dataframe(df)

heart = pd.read_csv("heart.csv")
X = heart.iloc[:,0:11].values
Y = heart.iloc[:,[11]].values

model = RandomForestClassifier()
model.fit(X,Y)

prediction = model.predict(df)
st.markdown('---')
st.subheader('Prediction :')
df1=pd.DataFrame(prediction,columns=['0'])
df1.loc[df1['0'] == 0, 'Chances of Heart Disease'] = 'No'
df1.loc[df1['0'] == 1, 'Chances of Heart Disease'] = 'Yes'
st.write(df1)

prediction_proba = model.predict_proba(df)
st.markdown('---')
st.subheader('Prediction Probability in % :')
st.write(prediction_proba * 100)
st.markdown('---')
st.subheader('Disclaimer : This is just a prediction using trained data and if you really want to know more about the health status of your heart just consult a doctor.')
st.subheader('THANK YOU!!')
st.markdown('---')
