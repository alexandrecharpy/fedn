FROM nvcr.io/nvidia/tensorflow:20.08-tf2-py3 AS nvidia-base

WORKDIR /app

FROM nvidia-base AS python-modules

RUN python3 -c 'import tensorflow as tf; print(tf.__version__)' && \
    pip install pandas scikit-learn keras

#    pip install keras && \

FROM python-modules AS scaleout-fedn

COPY sdk /app/sdk
COPY cli /app/cli

RUN pip install -e cli && \
    pip install -e sdk

