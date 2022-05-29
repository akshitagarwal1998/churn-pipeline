FROM python:3.7.13

COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN pip3 install -r requirements.txt
COPY . /opt/app

COPY training-script.py /usr/bin/train
COPY serve-script.py /usr/bin/serve

RUN chmod 755 /usr/bin/train /usr/bin/serve

EXPOSE 8080
