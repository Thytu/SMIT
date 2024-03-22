FROM python:3.10.13-slim as python-base

RUN apt-get update && \
    apt-get install -y build-essential libpq-dev gcc curl git-all && \
    apt-get remove -y && \
    apt-get autoremove -y && \
    apt-get clean -y

WORKDIR /app/

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY src/ src/