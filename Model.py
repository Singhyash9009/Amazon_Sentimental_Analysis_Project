# File Reading
import zipfile

# Array
import numpy as np
from numpy import mean,std

#DataFrame
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Vectorization
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier

# Model Building
from sklearn.model_selection import GridSearchCV,StratifiedKFold,KFold,train_test_split,cross_val_score,cross_val_predict
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier

#Model Evaluation
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,f1_score,precision_score
from sklearn.pipeline import Pipeline

# Time
from time import time
import streamlit as st
# Warnings
import warnings
warnings.filterwarnings('ignore')
@st.cache_data
def model():
    pantry_bkp=pd.read_csv('notebook/clean_data_finall.zip',compression='zip')
    #pantry_bkp.drop('Unnamed: 0',axis=1,inplace=True)
    pantry_bkp.head()

    # Dropping null values if exist any
    pantry_bkp.dropna(inplace=True)

    x=pantry_bkp['clean_text']
    y=pantry_bkp['Analysis']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

    # Label Encoding the classification data of target column
    ##########################################################################
    l_encoder=preprocessing.LabelEncoder()
    pantry_bkp['Analysis_encoded']=l_encoder.fit_transform(pantry_bkp['Analysis'])

    # x=pantry_bkp['clean_text']
    # y=pantry_bkp['Analysis_encoded']
    # x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

    vector = TfidfVectorizer()

    x_train_v = vector.fit_transform(x_train)
    x_test_v  = vector.transform(x_test)

    lr = LogisticRegression()
    # Penalty Type 
    penalty = ['l1', 'l2'] #l2 penalty- Ridge model, l1 penalty- Lasso model, It will used to reduce the error & increasing the accuracy of model

    # use logarithimically spaced c values 
    c= np.logspace(0, 4, 10) #Inverse of regularization(l1,l2) strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
    # Applying Grid Search CV
    grid_model_lr_tf = GridSearchCV(lr, 
                            param_grid = {'C':c, 'penalty': penalty})

    x=pantry_bkp['clean_text']
    y=pantry_bkp['Analysis']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

    vector = TfidfVectorizer()
    x_train = vector.fit_transform(x_train)
    x_test = vector.transform(x_test)
    #Classifing data of the four classes using a one vs. rest strategy with Logistic Regression
        

    #fitting training data into the model & predicting
        
    grid_model_lr_tf.fit(x_train, y_train)
        
    #y_pred = grid_model_lr_bow.predict(x_test)


    log=LogisticRegression(C= grid_model_lr_tf.best_params_['C'], penalty= grid_model_lr_tf.best_params_['penalty'])
    lr_tf_model=OneVsRestClassifier(log)
    lr_tf_model.fit(x_train,y_train)
    y_pred =lr_tf_model.predict(x_test)
    acc_trn=lr_tf_model.score(x_train,y_train)
    acc_tst=accuracy_score(y_test,y_pred)
    f1     =f1_score(y_test,y_pred,average='weighted')
    conf=confusion_matrix(y_test,y_pred)
    
    return lr_tf_model, vector


# Import necessary libraries for Streamlit
import streamlit as st

# Function to classify input text using the trained model and vectorizer

def classify_text(input_text, model, vector):
    # Vectorize the input text
    input_text_vectorized = vector.transform([input_text])
    
    # Predict using the trained model
    prediction = model.predict(input_text_vectorized)
    
    return prediction[0]

# Load the trained model and vectorizer
lr_tf_model, vector = model()

# Streamlit app
def main():
    st.title("Text Classification App")
    st.sidebar.header("User Input")

    # Get user input
    user_input = st.sidebar.text_area("Enter text for classification:", "")

    if st.sidebar.button("Classify"):
        # Ensure the user has entered some text
        if user_input:
            # Classify the text
            prediction = classify_text(user_input, lr_tf_model,vector)
            st.success(f"Predicted Class: {prediction}")
        else:
            st.warning("Please enter some text for classification.")

if __name__ == "__main__":
    main()
