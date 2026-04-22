"""
Change Detection Model — Siamese U-Net (TensorFlow/Keras).

Inputs:  image_t1 (256, 256, 4) + image_t2 (256, 256, 4)
Output:  (256, 256, 1) — Binary change mask (sigmoid)

Architecture:
  1. Shared Encoder (4 levels): processes t1 and t2 independently
  2. Feature Difference: diff = |F_t1 - F_t2| at each encoder level
  3. Decoder: U-Net style with skip connections from the diff features
  4. Output: 1×1 Conv with Sigmoid

Loss: Binary Crossentropy + Dice Loss
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def conv_block(x, filters, name_prefix):
    """Double convolution block: Conv→BN→ReLU × 2."""
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name_prefix}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_relu1")(x)

    x = layers.Conv2D(filters, 3, padding="same", name=f"{name_prefix}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_relu2")(x)

    return x


def build_shared_encoder(input_shape=(256, 256, 4)):
    """
    Build a shared encoder that returns multi-level features.

    Returns
    -------
    tf.keras.Model
        Encoder model with 5 outputs: [skip1, skip2, skip3, skip4, bottleneck]
    """
    inputs = layers.Input(shape=input_shape, name="encoder_input")

    # Level 1
    c1 = conv_block(inputs, 64, "shared_enc1")
    p1 = layers.MaxPooling2D((2, 2), name="shared_enc1_pool")(c1)

    # Level 2
    c2 = conv_block(p1, 128, "shared_enc2")
    p2 = layers.MaxPooling2D((2, 2), name="shared_enc2_pool")(c2)

    # Level 3
    c3 = conv_block(p2, 256, "shared_enc3")
    p3 = layers.MaxPooling2D((2, 2), name="shared_enc3_pool")(c3)

    # Level 4
    c4 = conv_block(p3, 512, "shared_enc4")
    p4 = layers.MaxPooling2D((2, 2), name="shared_enc4_pool")(c4)

    # Bottleneck
    bn = conv_block(p4, 1024, "shared_bottleneck")

    encoder = Model(inputs, [c1, c2, c3, c4, bn], name="SharedEncoder")
    return encoder


def build_siamese_unet(input_shape=(256, 256, 4)):
    """
    Build a Siamese U-Net for change detection.

    The encoder weights are shared between both inputs.
    Feature differences at each level are used as skip connections
    for the decoder.

    Parameters
    ----------
    input_shape : tuple
        Shape of each input image. Default: (256, 256, 4).

    Returns
    -------
    tf.keras.Model
        Siamese U-Net model with 2 inputs and 1 output.
    """
    # Shared encoder
    encoder = build_shared_encoder(input_shape)

    # Two inputs
    input_t1 = layers.Input(shape=input_shape, name="input_t1")
    input_t2 = layers.Input(shape=input_shape, name="input_t2")

    # Extract features from both inputs using the SAME encoder
    [skip1_t1, skip2_t1, skip3_t1, skip4_t1, bn_t1] = encoder(input_t1)
    [skip1_t2, skip2_t2, skip3_t2, skip4_t2, bn_t2] = encoder(input_t2)

    # Compute absolute differences at each level
    diff_bn   = layers.Lambda(lambda x: tf.abs(x[0] - x[1]), name="diff_bn")([bn_t1, bn_t2])
    diff_skip4 = layers.Lambda(lambda x: tf.abs(x[0] - x[1]), name="diff_skip4")([skip4_t1, skip4_t2])
    diff_skip3 = layers.Lambda(lambda x: tf.abs(x[0] - x[1]), name="diff_skip3")([skip3_t1, skip3_t2])
    diff_skip2 = layers.Lambda(lambda x: tf.abs(x[0] - x[1]), name="diff_skip2")([skip2_t1, skip2_t2])
    diff_skip1 = layers.Lambda(lambda x: tf.abs(x[0] - x[1]), name="diff_skip1")([skip1_t1, skip1_t2])

    # ---- DECODER (using diff features) ----

    # Level 4: 16→32
    d4 = layers.UpSampling2D((2, 2), name="dec4_up")(diff_bn)
    d4 = layers.Concatenate(name="dec4_concat")([d4, diff_skip4])
    d4 = conv_block(d4, 512, "dec4")

    # Level 3: 32→64
    d3 = layers.UpSampling2D((2, 2), name="dec3_up")(d4)
    d3 = layers.Concatenate(name="dec3_concat")([d3, diff_skip3])
    d3 = conv_block(d3, 256, "dec3")

    # Level 2: 64→128
    d2 = layers.UpSampling2D((2, 2), name="dec2_up")(d3)
    d2 = layers.Concatenate(name="dec2_concat")([d2, diff_skip2])
    d2 = conv_block(d2, 128, "dec2")

    # Level 1: 128→256
    d1 = layers.UpSampling2D((2, 2), name="dec1_up")(d2)
    d1 = layers.Concatenate(name="dec1_concat")([d1, diff_skip1])
    d1 = conv_block(d1, 64, "dec1")

    # ---- OUTPUT ----
    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="output")(d1)

    model = Model([input_t1, input_t2], outputs, name="Siamese_UNet_ChangeDetection")
    return model


def dice_loss(y_true, y_pred, smooth=1.0):
    """Dice loss for binary segmentation."""
    # Cast to float32 — required when using mixed precision (float16 predictions)
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice


def combined_loss(y_true, y_pred):
    """Binary Crossentropy + Dice Loss (both computed in float32)."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)
    d_loss = dice_loss(y_true, y_pred)
    return bce + d_loss


def iou_metric(y_true, y_pred):
    """Intersection-over-Union for binary segmentation."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.round(y_pred), tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + 1e-6) / (union + 1e-6)


def precision_metric(y_true, y_pred):
    """Pixel-wise precision."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.round(y_pred), tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    return (tp + 1e-6) / (tp + fp + 1e-6)


def recall_metric(y_true, y_pred):
    """Pixel-wise recall."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.round(y_pred), tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    return (tp + 1e-6) / (tp + fn + 1e-6)


def get_change_model(input_shape=(256, 256, 4), learning_rate=1e-4):
    """Build and compile the Siamese U-Net change detection model."""
    model = build_siamese_unet(input_shape)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=combined_loss,
        metrics=["accuracy", iou_metric, precision_metric, recall_metric],
    )
    return model


if __name__ == "__main__":
    # Quick test
    model = get_change_model()
    model.summary()

    # Test forward pass
    import numpy as np
    dummy_t1 = np.random.rand(2, 256, 256, 4).astype(np.float32)
    dummy_t2 = np.random.rand(2, 256, 256, 4).astype(np.float32)
    output = model.predict([dummy_t1, dummy_t2])
    print(f"Input shapes:  {dummy_t1.shape}, {dummy_t2.shape}")
    print(f"Output shape:  {output.shape}")
