# I2V-QLearning-AnimefaceGAN

## Setup

To set up the environment using Docker, follow these steps:
```bash
# Clone the repository
git clone --recursive https://github.com/LAI-YEN-CHUN/I2V-QLearning-AnimefaceGAN.git
cd I2V-QLearning-AnimefaceGAN
# Build and run the Docker container
docker-compose build
docker-compose up -d
# Download the pre-trained models
bash src/illustration2vec/get_models.sh
```

## References

- Chu, Shijie. "Leverage of Generative Adversarial Model for Boosting Exploration in Deep Reinforcement Learning." 2024 6th International Conference on Internet of Things, Automation and Artificial Intelligence (IoTAAI), 26 July 2024, pp. 315-318, ieeexplore.ieee.org/document/10692624, https://doi.org/10.1109/iotaai62601.2024.10692624. Accessed 28 Apr. 2025.