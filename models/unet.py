"""
Vegetation Detection Model — 4-Level U-Net (TensorFlow/Keras).

Input:  (256, 256, 4) — Sentinel-2 bands [B2, B3, B4, B8]
Output: (256, 256, 1) — Binary vegetation mask (sigmoid)

Architecture:
  Encoder:  4 levels (64 → 128 → 256 → 512) with Conv→BN→ReLU→MaxPool
  Bottleneck: 1024 filters
  Decoder:  4 levels with UpSampling→Concat(skip)→Conv→BN→ReLU
  Output:   1×1 Conv with Sigmoid

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


def encoder_block(x, filters, name_prefix):
    """Encoder block: double conv + max pool. Returns (skip, pooled)."""
    skip = conv_block(x, filters, name_prefix)
    pooled = layers.MaxPooling2D(pool_size=(2, 2), name=f"{name_prefix}_pool")(skip)
    return skip, pooled


def decoder_block(x, skip, filters, name_prefix):
    """Decoder block: upsample + concat skip + double conv."""
    x = layers.UpSampling2D(size=(2, 2), name=f"{name_prefix}_up")(x)
    x = layers.Concatenate(name=f"{name_prefix}_concat")([x, skip])
    x = conv_block(x, filters, name_prefix)
    return x


def build_unet(input_shape=(256, 256, 4)):
    """
    Build a 4-level U-Net for vegetation segmentation.

    Parameters
    ----------
    input_shape : tuple
        Input image shape (H, W, C). Default: (256, 256, 4).

    Returns
    -------
    tf.keras.Model
        Compiled U-Net model.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # ---- ENCODER ----
    skip1, p1 = encoder_block(inputs, 64,  "enc1")   # 256→128
    skip2, p2 = encoder_block(p1,     128, "enc2")   # 128→64
    skip3, p3 = encoder_block(p2,     256, "enc3")   # 64→32
    skip4, p4 = encoder_block(p3,     512, "enc4")   # 32→16

    # ---- BOTTLENECK ----
    bottleneck = conv_block(p4, 1024, "bottleneck")    # 16×16×1024

    # ---- DECODER ----
    d4 = decoder_block(bottleneck, skip4, 512, "dec4")  # 16→32
    d3 = decoder_block(d4,        skip3, 256, "dec3")  # 32→64
    d2 = decoder_block(d3,        skip2, 128, "dec2")  # 64→128
    d1 = decoder_block(d2,        skip1, 64,  "dec1")  # 128→256

    # ---- OUTPUT ----
    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="output")(d1)

    model = Model(inputs, outputs, name="UNet_Vegetation")

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


def get_vegetation_model(input_shape=(256, 256, 4), learning_rate=1e-4):
    """Build and compile the vegetation U-Net model."""
    model = build_unet(input_shape)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=combined_loss,
        metrics=["accuracy", iou_metric, precision_metric, recall_metric],
    )
    return model


if __name__ == "__main__":
    # Quick test
    model = get_vegetation_model()
    model.summary()

    # Test forward pass
    import numpy as np
    dummy = np.random.rand(2, 256, 256, 4).astype(np.float32)
    output = model.predict(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {output.shape}")
