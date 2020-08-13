FROM ubuntu:18.04

ENV DEBIAN_FRONTEND noninteractive.

RUN apt-get update && apt-get install -y python3 \
        python3-dev \
        python3-pip \
        git \
        swig && \
    apt-get upgrade -y && apt-get install -y language-pack-ru

ENV LANGUAGE ru_RU.UTF-8
ENV LANG ru_RU.UTF-8
ENV LC_ALL ru_RU.UTF-8
RUN locale-gen en_US en_US.UTF-8 && \
    locale-gen ru_RU ru_RU.UTF-8 && \
    dpkg-reconfigure locales && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt

RUN pip3 install --no-cache-dir pytest-runner \
    && pip3 install --no-cache-dir -r requirements.txt

ENV RESOURCESDIR ./resources

RUN apt-get update && apt-get install unzip

COPY ./ ./

RUN python3 -c "from deeppavlov import build_model, configs; tokenizer, elmo = build_model(configs.embedder.elmo_ru_twitter, download=True)"

ENTRYPOINT python3 app.py 8000
