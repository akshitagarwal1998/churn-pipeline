FROM ubuntu:bionic
MAINTAINER Akshit Agarwal "akshitagarwal1998@gmail.com"

RUN apt-get update && apt-get upgrade -y

FROM python:3.7.13

COPY requirements.txt  /usr/app/src/requirements.txt
COPY attributes.txt  /usr/app/src/attributes.txt
COPY training-script.py  /usr/app/src/training-script.py
COPY serve-script.py  /usr/app/src/serve-script.py
COPY data.csv  /usr/app/src/data.csv
COPY serve-model.pkl  /usr/app/src/serve-model.pkl
COPY input_data.csv  /usr/app/src/input_data.csv

CMD echo "This is a test."

RUN chmod 777  /usr/app/src

WORKDIR  /usr/app/src
RUN pip3 install -r requirements.txt
RUN pip3 install boto3
COPY .  /usr/app/src

CMD [ "python3", "serve-script.py" ]

EXPOSE 8080
