FROM python:3.12

WORKDIR /Het_CD

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY ./source ./src