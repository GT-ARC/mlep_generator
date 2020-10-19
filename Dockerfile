FROM mxnet/python:1.5.0_gpu_cu101_mkl_py3

WORKDIR /app

#Setup software repositories and update base system
RUN apt-get update && apt-get install -y \
    software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa

#This is required by pip to beeing able to read UTF-8 packages reaquired by SMAC
RUN apt-get update && apt-get install -y locales language-pack-en nano build-essential swig cmake git
RUN locale-gen en_US.UTF-8
ENV LANG C.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

#Install python>=3.7
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p /miniconda && \
    echo "export PATH="/miniconda/bin:$PATH"" >> ~/.bashrc && \
    /bin/bash -c "source ~/.bashrc"
ENV PATH=/miniconda/bin:${PATH}

#add the local directory to the /app directory to make it available for the following pip install requirements.txt call
ADD . /app

RUN conda install -c anaconda setuptools
RUN conda update -y conda
RUN conda create -y --name as python=3.7 pip
RUN echo "source activate as" > ~/.bashrc
RUN /miniconda/envs/as/bin/pip install --upgrade pip
RUN /miniconda/envs/as/bin/pip install --trusted-host pypi.python.org -r requirements.txt


ENV PATH /miniconda/envs/as/bin:$PATH

ENTRYPOINT ["python", "run.py"]