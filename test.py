import tensorflow as tf

import display
from config import CONFIG
from dataset import get_data
from model import mobile_net_x_unet

if __name__ == "__main__":
    train_data, test_data, _ = get_data(training=True)
    trained = True
    if trained:
        model = tf.keras.models.load_model(CONFIG.MODEL_PATH)
    else:
        model = mobile_net_x_unet()
    display.show_predictions(dataset=test_data, model=model, num=3)
