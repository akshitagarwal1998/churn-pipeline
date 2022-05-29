import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'white')
import os
import imblearn
import glob
import logging
import pickle as pkl

logging.basicConfig(level=logging.INFO,filename='train-pipeline.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logging.info("curdir: %s",os.getcwd())

# Standard plotly imports
import chart_studio.plotly as py
import plotly.graph_objects as go
import plotly.tools as tls
from plotly.offline import iplot, init_notebook_mode
import plotly.figure_factory as ff

# Preprocessing Imports
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

#Datetime Imports
import datetime
from datetime import datetime, timedelta,date
from dateutil.relativedelta import *

# boto3
import boto3
# s3_client = boto3.resource('s3')

# Model Definitons
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold,cross_val_score,StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
from xgboost import XGBClassifier

def dataset_summary(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values
    return summary

def convert_tenure_quarters(df,col="tenure"):
  def subtract_months(x):
    return (reference_date - relativedelta(months=+int(x)))

  df2 = df[col].apply(subtract_months)
  df2 = pd.to_datetime(df2)
  df2 = pd.PeriodIndex(df2, freq='Q')
  return df2.astype(str)

if __name__ == "__main__":
    # Get S3 bucket and use the data for training
    # load the model from S3 location
    try:
        input_path = './train_data.csv' # Can be accessed by s3 location
        df = pd.read_csv(input_path)
        logging.info('Input Data Shape: (%d,%d)',df.shape[0],df.shape[1])
    except Exception as e:
        logging.error("Input Data Not Found",exc_info=True)
        exit()

    #Time Column - Quarterly
    max_tenure = df['tenure'].max()
    reference_date = date(2022,5,31)
    startdate = reference_date - relativedelta(months=+max_tenure)
    enddate = reference_date
    date_range = pd.period_range(start=startdate, end=enddate, freq='Q')

    df["Tenure"] = convert_tenure_quarters(df,"tenure")

    df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='coerce')
    df.TotalCharges.isnull().sum()
    df.drop(["customerID"],axis=1,inplace=True)
    # #Converting the target variable in to binary numeric variable
    df['Churn'].replace(to_replace='Yes', value=1, inplace=True)
    df['Churn'].replace(to_replace='No',  value=0, inplace=True)

    #LabelEncover the df
    df_le = df.apply(LabelEncoder().fit_transform)
    # Performin MinMaxScaing
    sc = MinMaxScaler()
    df_le_sc = pd.DataFrame(sc.fit_transform(df_le.values), columns=df_le.columns, index=df_le.index)
    # Dump the scaler to use in testing
    pkl.dump(sc,open("sc.pkl","wb"))

    # Class Balance using SMOTE
    oversample = SMOTE()
    y=df_le_sc['Churn'].values
    X = df_le_sc.drop(columns = ['Churn'])

    logging.info("Number of Data points before Oversampling, %s",str(Counter(y)))
    X, y = oversample.fit_resample(X,y)
    logging.info("Over Sampling done by SMOTE, %s",str(Counter(y)))
    # Create Train & Test Data based on the STRATIFICATION
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42,stratify=y)

    # Initialze the estimators
    clf1 = RandomForestClassifier(random_state=42)
    clf2 = SVC(probability=True, random_state=42)
    clf3 = LogisticRegression(random_state=42)
    clf4 = DecisionTreeClassifier(random_state=42)
    clf5 = KNeighborsClassifier()
    clf6 = MultinomialNB()
    clf7 = GradientBoostingClassifier(random_state=42)
    clf8 = XGBClassifier(random_state=42)

    #Defining the Stratified K Fold Validation for GridSearch And RandomSearch
    skf = StratifiedKFold(n_splits=10,random_state=42,shuffle=True)

    # Initiaze the hyperparameters for each dictionary
    param1 = {}
    param1['classifier__n_estimators'] = [10, 50, 100, 250,500]
    param1['classifier__max_depth'] = [5, 10, 20]
    param1['classifier__class_weight'] = [None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}]
    param1['classifier'] = [clf1]

    param2 = {}
    param2['classifier__C'] = [10**-2, 10**-1, 10**0, 10**1, 10**2]
    param2['classifier__class_weight'] = [None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}]
    param2['classifier'] = [clf2]

    param3 = {}
    param3['classifier__C'] = [10**-2, 10**-1, 10**0, 10**1, 10**2]
    param3['classifier__penalty'] = ['l1', 'l2']
    param3['classifier__class_weight'] = [None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}]
    param3['classifier'] = [clf3]

    param4 = {}
    param4['classifier__max_depth'] = [5,10,25,None]
    param4['classifier__min_samples_split'] = [2,5,10]
    param4['classifier__class_weight'] = [None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}]
    param4['classifier'] = [clf4]

    param5 = {}
    param5['classifier__n_neighbors'] = [2,5,10,25,50]
    param5['classifier'] = [clf5]

    param6 = {}
    param6['classifier__alpha'] = [10**0, 10**1, 10**2]
    param6['classifier'] = [clf6]

    param7 = {}
    param7['classifier__n_estimators'] = [10, 50, 100, 250]
    param7['classifier__max_depth'] = [5, 10, 20]
    param7['classifier'] = [clf7]

    param8 = {}
    param8['classifier__min_child_weight'] = [1,5,10,]
    param8['classifier__gamma']= [0.5, 1, 1.5, 2, 5]
    param8['classifier__subsample']= [0.6, 0.8, 1.0]
    param8['classifier__colsample_bytree']= [ 0.3, 0.4, 0.5 , 0.7 ]
    param8['classifier__max_depth']= [3, 4, 5,6,7,8,9,10]
    param8['classifier__learning_rate'] =  [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ]
    param8['classifier'] = [clf8]

    # Definig Pipeline
    pipeline = Pipeline([('classifier', clf1)])
    params = [param1, param2, param3, param4, param5, param6, param7,param8]
    logging.info("#%d Model being Trained",len(params))

    # Grid Search CV
    gs = GridSearchCV(pipeline, params, cv=skf, n_jobs=-1, scoring='roc_auc').fit(X_train, y_train)
    logging.info("Grid Search Best Params: %s",gs.best_params_)
    logging.info("Grid Search ROC-AUC Score: %d",gs.best_score_)

    # Test data performance
    logging.info("Test Precision %d",precision_score(gs.predict(X_test), y_test))
    logging.info("Test Recall %d",recall_score(gs.predict(X_test), y_test))
    logging.info("Test ROC AUC Score %d",roc_auc_score(gs.predict(X_test), y_test))

    # Random Search CV
    rs = RandomizedSearchCV(pipeline, params, cv=skf, n_jobs=-1, scoring='roc_auc').fit(X_train, y_train)
    logging.info("Random Search Best Params: %s",rs.best_params_)
    logging.info("Random Search ROC-AUC Score: %d",rs.best_score_)

    # Test data performance
    logging.info("Test Precision %d",precision_score(rs.predict(X_test), y_test))
    logging.info("Test Recall %d",recall_score(rs.predict(X_test), y_test))
    logging.info("Test ROC AUC Score %d",roc_auc_score(rs.predict(X_test), y_test))

    # Saving the best model
    model = gs
    if gs.best_score_ < rs.best_score_:
        model = rs
    model_path = "serve-model.pkl"
    pkl.dump(model,open(model_path,"wb"))
