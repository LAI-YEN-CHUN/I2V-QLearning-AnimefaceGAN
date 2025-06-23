# I2V-QLearning-AnimefaceGAN

## Setup

To set up the environment using Docker and Docker Compose, follow these steps:
```bash
# Clone the repository
git clone --recursive https://github.com/LAI-YEN-CHUN/I2V-QLearning-AnimefaceGAN.git
cd I2V-QLearning-AnimefaceGAN
# Build and run the Docker container
docker-compose build
docker-compose up -d
```

## Usage

Once the Docker container is running, you can execute the following commands to download the pre-trained models and run the main script:
```bash
# Download the pre-trained models
bash src/illustration2vec/get_models.sh
# Execute the main script
python src/main.py
```

## References

- Chu, Shijie. "Leverage of Generative Adversarial Model for Boosting Exploration in Deep Reinforcement Learning." 2024 6th International Conference on Internet of Things, Automation and Artificial Intelligence (IoTAAI), 26 July 2024, pp. 315-318, ieeexplore.ieee.org/document/10692624, https://doi.org/10.1109/iotaai62601.2024.10692624. Accessed 28 Apr. 2025.