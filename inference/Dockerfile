FROM nvcr.io/nvidia/pytorch:24.01-py3

ARG NVIDIA_SM=90

RUN apt-get update -y

RUN pip install --upgrade pip virtualenv
RUN pip config unset global.extra-index-url

RUN addgroup arctic -gid 1000
RUN adduser -disabled-password -u 1000 -gid 1000 arctic

RUN chown arctic:arctic /home/arctic

USER arctic
WORKDIR /home/arctic

RUN virtualenv venv

RUN source venv/bin/activate && \
    pip install git+https://github.com/Snowflake-Labs/vllm.git@arctic && \
    pip install git+https://github.com/Snowflake-Labs/transformers.git@arctic && \
    pip install deepspeed>=0.14.2
