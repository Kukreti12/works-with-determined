FROM determinedai/environments:cuda-10.0-pytorch-1.4-tf-1.15-gpu-0.7.0 as base
COPY environment.yml /tmp/
RUN conda --version && \
   conda env update --name base --file /tmp/environment.yml && \
   conda clean --all --force-pkgs-dirs --yes
RUN eval "$(conda shell.bash hook)" && conda activate base