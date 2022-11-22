# Determined Image
FROM determinedai/environments:py-3.8-pytorch-1.10-tf-2.8-cpu-0.19.4

# Create a virtual environment
COPY pip_requirements.txt /tmp/pip_requirements.txt
RUN conda create -n myenv python=3.8
RUN eval "$(conda shell.bash hook)" && \
   conda activate myenv && \
   pip install -r /tmp/pip_requirements.txt

# Set the default virtual environment
RUN echo 'eval "$(conda shell.bash hook)" && conda activate myenv' >> ~/.bashrc