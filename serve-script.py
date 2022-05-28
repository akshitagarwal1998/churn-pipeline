#!/usr/bin/env python
import os
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import pickle
import logging
# import boto3
# import botocore

logging.basicConfig(level=logging.INFO,filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logging.info("curdir: %s",os.getcwd())

if __name__ == "__main__":
    # model path S3 locaiton
    model_path = 'serve-model.pkl'
    logging.info('Model Path S3:%s ', model_path)
    # load the model from S3 location
    try:
        model = pickle.load(open(model_path, 'rb'))
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


    for line in lines:
        line = line.split("\n")
        attributes_list.append(line[0])

    logging.info('%d Attributes being used', len(attributes_list))

    # Input Attributes in CSV Format
    '''
    S3 Location or POST Request for the input data
    '''

    model_input = 'input_data.csv'
    logging.info('Model Input S3:%s ', model_input)
    try:
        df_test = pd.read_csv(model_input)
    except Exception as e:
        logging.error("Test Inputs are missing!!",exc_info=True)
        exit()


    #Selecting the limited columsn for testing
    input_data = df_test[df_test.columns.intersection(attributes_list)]
    '''
    Pre-Process the test input_data using Standardization/Normalization based on the train data corresponding to the model
    '''

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
