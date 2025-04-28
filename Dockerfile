# Use the official NVIDIA PyTorch image as a base
FROM nvcr.io/nvidia/pytorch:24.11-py3
# Set the working directory
WORKDIR /workspace
COPY . .

# Install the required packages
RUN pip install --upgrade pip
RUN pip install notebook ipywidgets pylatexenc
RUN pip install -r requirements.txt