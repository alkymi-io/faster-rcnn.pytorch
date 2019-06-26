FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime

RUN apt-get update
RUN apt-get install -y python-qt4 \
    poppler-utils \
    wget \
    nginx \
    ca-certificates

COPY ./ /opt/program/
COPY ./hyperparameters.json /opt/ml/input/config/hyperparameters.json

RUN pip install -r /opt/program/requirements.txt
WORKDIR /opt/program/lib

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

WORKDIR /opt/program/
