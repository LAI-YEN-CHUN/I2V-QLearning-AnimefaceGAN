# I2V-QLearning-AnimefaceGAN

- I2V-QLearning-AnimefaceGAN is an implementation based on the following paper: *"Auxiliary Generative Adversarial Networks with Iliustration2Vec and Q-Learning based Hyperparameter Optimisation for Anime Image Synthesis"* by Vivian Sedov and Li Zhang.
- This project integrates Generative Adversarial Networks (GANs) with Q-Learning to enhance exploration in anime face generation tasks.
- This repository provides a reproducible environment using Docker and Docker Compose, ensuring consistent results across different systems.

## Prerequisites

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/)
- (Optional) [Visual Studio Code](https://code.visualstudio.com/) with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) for seamless development inside containers

## Quick Start

### 1. Clone the Repository

```bash
git clone --recursive https://github.com/LAI-YEN-CHUN/I2V-QLearning-AnimefaceGAN.git
cd I2V-QLearning-AnimefaceGAN
```

### 2. Build and Run with Docker Compose

```bash
docker-compose build
docker-compose up -d
```

This will build the required Docker image and start the container in detached mode.

### 3. (Optional) Open in VSCode Dev Container

If you use Visual Studio Code, you can leverage the Dev Containers extension for a streamlined development experience:

1. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).
2. Open the project folder in VSCode.
3. Press `F1` and select `Dev Containers: Reopen in Container`.

This will automatically build and open the workspace inside the Docker container.

## Usage

Once the Docker container is running, you can execute the following commands inside the container:

```bash
# Download the pre-trained models
bash src/illustration2vec/get_models.sh

# Run the main script
python src/main.py
```

## Citation

- Chu, Shijie. "Leverage of Generative Adversarial Model for Boosting Exploration in Deep Reinforcement Learning." 2024 6th International Conference on Internet of Things, Automation and Artificial Intelligence (IoTAAI), 26 July 2024, pp. 315-318, ieeexplore.ieee.org/document/10692624, https://doi.org/10.1109/iotaai62601.2024.10692624. Accessed 28 Apr. 2025.
