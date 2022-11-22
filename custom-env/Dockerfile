# Determined Image
FROM determinedai/environments:cuda-11.3-pytorch-1.10-tf-2.8-gpu-0.19.4

# Custom Configuration
RUN apt-get update && \
   DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
COPY pip_requirements.txt /tmp/pip_requirements.txt
RUN conda clean --all --force-pkgs-dirs --yes
RUN eval "$(conda shell.bash hook)" && \
   conda activate base && \
   pip install --requirement /tmp/pip_requirements.txt
