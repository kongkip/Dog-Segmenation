import wandb
from wandb.keras import WandbCallback

from config import CONFIG
from train import run

wandb.init(project='DogSegmentation', entity='kongkip')

wandb_config = wandb.config

wandb_config.learning_rate = CONFIG.LEARNING_RATE
wandb_config.epochs = CONFIG.EPOCHS
wandb_config.buffer_size = CONFIG.BUFFER_SIZE
wandb_config.batch_size = CONFIG.BATCH_SIZE
wandb_config.image_size = CONFIG.IMAGE_SIZE

callbacks = [WandbCallback()]

if __name__ == "__main__":
    run(callbacks)
