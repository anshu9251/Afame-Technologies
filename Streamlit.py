import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# import the dataset

df = pd.read_csv("D:/creditcard.csv")
print(df.head())

# Separate the data into fraud and legit

legit = df[df.Class==0]
fraud = df[df.Class==1]

# Performing UnderSampling to deal with imbalance data

legit_sample = legit.sample(n=473)
df_new = pd.concat([legit_sample,fraud],axis=0)

# Splitting our dataset into features and target

X = df_new.drop(columns="Class",axis=1)
y = df_new["Class"]

# Performing TrainTest split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=2)

# Model Building
etc = ExtraTreesClassifier(n_estimators=100)
etc.fit(X_train,y_train)
y_pred = etc.predict(X_test)

print(accuracy_score(y_pred,y_test))

# Web app

st.title("Credit Card Fraud Detection Model")
input_val = st.text_area("Enter the transaction values")

input_df_splited = input_val.split(",")
predict = st.button("predict")

if predict:
    features = np.asarray(input_df_splited,dtype=np.float64)
    predictions = etc.predict(features.reshape(1,-1))

    if predictions[0]==0:
        st.write("Legit Transaction")
    else:
        st.write("Fraud transaction")