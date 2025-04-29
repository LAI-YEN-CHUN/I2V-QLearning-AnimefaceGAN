# Use the official NVIDIA PyTorch image as a base
FROM nvcr.io/nvidia/pytorch:23.12-py3

# Set the working directory
WORKDIR /workspace
COPY . .

# Set the environment variable to avoid cross-platform git autocrlf issues
RUN git config core.autocrlf input

# Install the required packages
RUN pip install --upgrade pip
RUN pip install notebook ipywidgets pylatexenc
RUN pip install -r requirements.txt