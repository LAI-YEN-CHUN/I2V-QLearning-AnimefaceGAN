# Configuration factory for dataset preprocessing and model paths
import os

import torch


class BaseConfig:
    """
    Base configuration class for all global constants and paths.
    """

    def __init__(self):
        # Root directory of the project
        self.ROOT_DIR = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../../')
        )
        self.DATA_DIR = os.path.join(self.ROOT_DIR, 'data')
        self.RESULTS_DIR = os.path.join(self.ROOT_DIR, 'results')
        self.I2V_MODULE_DIR = os.path.join(
            self.ROOT_DIR, 'src/illustration2vec')
        self.FID_REAL_IMAGES_DIR = os.path.join(
            self.RESULTS_DIR, 'metrics/real_images')
        self.FID_FAKE_IMAGES_DIR = os.path.join(
            self.RESULTS_DIR, 'metrics/fake_images')

        # Dataset directories
        self.RAW_DATA_DIR = os.path.join(self.DATA_DIR, 'raw')
        self.RAW_IMAGES_DIR = os.path.join(self.RAW_DATA_DIR, 'images')
        self.PROCESSED_DATA_DIR = os.path.join(self.DATA_DIR, 'processed')
        self.PROCESSED_IMAGES_DIR = os.path.join(
            self.PROCESSED_DATA_DIR, 'images')
        self.LABELS_CSV = os.path.join(self.PROCESSED_DATA_DIR, 'labels.csv')

        # Results directories
        self.GENERATOR_CHECKPOINTS_DIR = os.path.join(
            self.RESULTS_DIR, 'checkpoints/generator')
        self.DISCRIMINATOR_CHECKPOINTS_DIR = os.path.join(
            self.RESULTS_DIR, 'checkpoints/discriminator')
        self.SAMPLES_DIR = os.path.join(self.RESULTS_DIR, 'samples')
        self.LOGS_DIR = os.path.join(self.RESULTS_DIR, 'logs')
        self.FIGURES_DIR = os.path.join(self.RESULTS_DIR, 'figures')

        # Illustration2Vec model paths
        self.I2V_CAFFEMODEL_PATH = os.path.join(
            self.I2V_MODULE_DIR, 'illust2vec_tag_ver200.caffemodel')
        self.I2V_TAG_LIST_PATH = os.path.join(
            self.I2V_MODULE_DIR, 'tag_list.json')

        # 22 target tags for attribute extraction
        self.TARGET_TAGS = [
            '1girl',
            '1boy',
            'blue hair',
            'green hair',
            'red hair',
            'black hair',
            'pink hair',
            'orange hair',
            'purple hair',
            'brown hair',
            'aqua hair',
            'white hair',
            'blue eyes',
            'green eyes',
            'red eyes',
            'black eyes',
            'pink eyes',
            'orange eyes',
            'purple eyes',
            'brown eyes',
            'aqua eyes',
            'yellow eyes',
        ]
        # TARGET_TAGS: [0-1]: sex, [2-11]: hair color, [12-21]: eye color
        self.NUM_SEX = 2
        self.NUM_HAIR_COLORS = 10
        self.NUM_EYE_COLORS = 10

        # Training Device
        self.DEVICE = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png']

        # Hyperparameters
        self.RANDOM_SEED = 42
        self.NOISE_DIM = 100
        self.CLASS_DIM = len(self.TARGET_TAGS)
        self.IMAGE_SIZE = 64
        self.BATCH_SIZE = 128
        self.EPOCHES = 100
        self.CRITICS = 1
        # self.ADAM_BETA_1 = 0.5
        # self.ADAM_BETA_2 = 0.5
        self.LAMBDA_CLASS = 1.0
        self.LAMBDA_GP = 10.0
        self.NUM_TEST_IMAGES = 10000

        self.SAMPLE_INTERVAL = 1
        self.MODEL_SAVE_INTERVAL = 10

        # Initialization Hyperparameters
        self.INIT_LEARNING_RATE = 0.001
        self.INIT_MOMENTUM = 0.9
        self.INIT_WEIGHT_DECAY = 0.0005
