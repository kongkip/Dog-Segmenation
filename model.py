import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_examples.models.pix2pix import pix2pix

from config import CONFIG


def mobile_net_x_unet():
    input_shape = CONFIG.IMAGE_SIZE + [3, ]
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False)

    # activation layers to use
    layer_names = [
        'block_1_expand_relu',  # 64x64
        'block_3_expand_relu',  # 32x32
        'block_6_expand_relu',  # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',  # 4x4
    ]

    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.inputs, outputs=base_model_outputs)

    down_stack.trainable = False

    # upsampling layers
    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),  # 32x32 -> 64x64
    ]

    # define unet_mode
    inputs = layers.Input(shape=input_shape)

    # downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # upsampling and establishing skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = layers.Concatenate()
        x = concat([x, skip])

    last = layers.Conv2DTranspose(3, 3, strides=2, padding="same")

    x = last(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    # unet(config).summary()
    mobile_net_x_unet().summary()
