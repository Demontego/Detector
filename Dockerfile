FROM ubuntu:22.04

WORKDIR /server


RUN apt-get update && apt-get -y upgrade \
    && apt-get install -y --no-install-recommends \
    git \
    wget \
    g++ \
    gcc \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*


ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
COPY ./environment.yaml environment.yaml
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
	&& echo "Running $(conda --version)" && \
    conda init bash && \
	. /root/.bashrc && \
	conda update conda &&\
    conda env create -f environment.yaml && \
    conda activate detector && \
    pip install sahi==0.11.0
    # pip install icevision effdet lightning-flash[image] -U  &&\
    # pip install sahi==0.11.0 && \
    # pip install lightning-flash[serve]


COPY ./detector.py detector.py
# COPY ./requirements.txt requirements.txt

# RUN conda env create -f environment.yaml
# SHELL ["conda", "run", "-n", "detector", "/bin/bash", "-c"]
# RUN conda init 
# RUN conda activate detector

# RUN conda activate sbercloud
COPY ./object_detection_model_1epoch.pt object_detection_model_1epoch.pt

EXPOSE 8000

# ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
# CMD ["python", "detector.py"]
ENTRYPOINT ["conda", "run",  "-n", "detector", "python", "detector.py"]