import tensorflow as tf
import tensorflow_datasets as tfds

from config import CONFIG
from display import display

dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


@tf.function
def load_image_train(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def load_image_test(datapoint):
    input_image = tf.image.resize(datapoint['image'], CONFIG.IMAGE_SIZE)
    input_mask = tf.image.resize(datapoint['segmentation_mask'], CONFIG.IMAGE_SIZE)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def get_data(training=False):
    train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
    test = dataset['test'].map(load_image_test)

    if training:
        print("INFO: Getting prepared data")
        train_dataset = train.cache().shuffle(CONFIG.BUFFER_SIZE).batch(CONFIG.BATCH_SIZE).repeat()
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        test_dataset = test.batch(CONFIG.BATCH_SIZE)

        return train_dataset, test_dataset, info
    return train, test


if __name__ == "__main__":
    sample_train, sample_test = get_data(training=False)

    sample_image, sample_mask = (None, None)

    for image, mask in sample_train.take(1):
        print(image.shape)
        print(mask.shape)
        sample_image, sample_mask = image, mask
    display([sample_image, sample_mask])
