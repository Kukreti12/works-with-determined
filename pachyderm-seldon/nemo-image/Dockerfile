# Determined Image
FROM determinedai/environments:cuda-11.3-pytorch-1.10-tf-2.8-gpu-0.19.4

WORKDIR /app
COPY ./* /app/
RUN apt-get update && \
   DEBIAN_FRONTEND="noninteractive" apt-get -y install unzip libsndfile1 ffmpeg
# Custom Configuration
RUN conda create -n nemo python=3.9
RUN eval "$(conda shell.bash hook)" && \
   conda activate nemo
RUN git clone -q --depth 1 https://github.com/NVIDIA/apex /app/apex
RUN cd /app/apex && pip install --no-cache-dir /app/apex
RUN pip install wget unidecode sox && \
   pip install Cython && \
   pip install nemo-toolkit['all']
   
RUN echo 'eval "$(conda shell.bash hook)" && conda activate nemo' >> ~/.bashrc 
ENV PYTHONUNBUFFERED=1