from dataclasses import dataclass


@dataclass
class CONFIG:
    EPOCHS = 20
    LEARNING_RATE = 0.01
    BUFFER_SIZE = 1000
    BATCH_SIZE = 32
    IMAGE_SIZE = [128, 128]
    MODEL_PATH = "models/mobile_net_x_unet.h5"
