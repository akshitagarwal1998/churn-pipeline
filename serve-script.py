#!/usr/bin/env python
import os
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
import pandas as pd
import pickle
import logging
import boto3
import datetime
from datetime import datetime, timedelta,date
from dateutil.relativedelta import *

logging.basicConfig(level=logging.INFO,filename='serve-pipeline.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logging.info("curdir: %s",os.getcwd())

def convert_tenure_quarters(df,col="tenure"):
  reference_date = date(2022,5,31)
  def subtract_months(x):
    return (reference_date - relativedelta(months=+int(x)))

  df2 = df[col].apply(subtract_months)
  df3 = pd.to_datetime(df2)
  df4 = pd.PeriodIndex(df3, freq='Q')
  return df4.astype(str)

if __name__ == "__main__":


    # # S3 client
    # s3 = boto3.client("s3")
    # s3.download_file(
    # Bucket="churn-pred-aksagarw", Key="serve-model.pkl", Filename="serve-model.pkl"
    # )

    # model path S3 locaiton
    model_path = 'serve-model.pkl'
    sc_path = 'sc.pkl'
    logging.info('Model Path S3:%s ', model_path)
    # load the model from S3 location
    try:
        model = pickle.load(open(model_path, 'rb'))
        sc = pickle.load(open(sc_path,'rb'))
    except Exception as e:
        logging.error("Model Not Found",exc_info=True)
        exit()

    #prediction attributes s3 location
    model_attributes_path = 'attributes.txt'
    logging.info('Attributes Path S3:%s ', model_attributes_path)
    attributes_list = []

    try:
        with open(model_attributes_path,"r") as f:
            lines = f.readlines()
    except Exception as e:
        logging.error("Model Attributes Not Found",exc_info=True)
        exit()

    logging.info("Attribute Used:")
    for ix,line in enumerate(lines):
        line = line.split("\n")
        attributes_list.append(line[0])
        logging.info("%d %s",ix+1,line[0])

    logging.info('Total %d Attributes being used', len(attributes_list))

    # Input Attributes in CSV Format
    '''
    S3 Location or POST Request for the input data
    '''

    model_input = 'test_Data.csv'
    logging.info('Model Input S3:%s ', model_input)
    try:
        df_test = pd.read_csv(model_input)
        df_test["TenureInQuarters"] = convert_tenure_quarters(df_test,"tenure")
    except Exception as e:
        logging.error("Test Inputs are missing!!",exc_info=True)
        exit()

    # Pre-Process the test input_data using Standardization/Normalization based on the train data corresponding to the model
    df_test = df_test.apply(LabelEncoder().fit_transform)
    # input_data = pd.DataFrame(sc.transform(df_test.values), columns=df_test.columns, index=df_test.index)
    #Selecting the limited columsn for testing
    input_data = df_test[attributes_list]

    #Predict on the input_data
    try:
        results = model.predict(input_data)
        logging.info("#%d Predictions Done",len(input_data))
        results = pd.Series(results)
        results.name = "Churn"

        #Save the results to S3 Bucket
        savepath = "results.csv"
        results.to_csv(savepath, index=False)
        logging.info("Results Written to S3: %s",savepath)
    except Exception as e:
        logging.critical("Prediction Error Occured",exc_info=True)
