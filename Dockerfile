FROM pytorch/pytorch:0.4-cuda9-cudnn7-devel

RUN apt-get update
RUN apt-get install -y python-qt4 \
    poppler-utils \
    wget \
    nginx \
    ca-certificates

COPY ./ /opt/program/

RUN pip install -r /opt/program/requirements.txt
WORKDIR /opt/program/lib

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

WORKDIR /opt/program/
