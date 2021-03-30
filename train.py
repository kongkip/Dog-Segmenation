import tensorflow as tf
from IPython.display import clear_output

import display
from config import CONFIG
from dataset import get_data
from model import mobile_net_x_unet


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        # show_predictions()


def run(callbacks: list):
    callbacks.append(DisplayCallback())
    train_data, test_data, info = get_data(training=True)

    model = mobile_net_x_unet()
    steps_per_epoch = info.splits["train"].num_examples // CONFIG.BATCH_SIZE
    validation_steps = info.splits["test"].num_examples // CONFIG.BATCH_SIZE

    model.fit(train_data, epochs=CONFIG.EPOCHS,
              steps_per_epoch=steps_per_epoch,
              validation_data=test_data,
              validation_steps=validation_steps,
              callbacks=callbacks)
    model.save("models/mobile_net_x_unet.h5")
    display.show_predictions(test_data, model=model)
