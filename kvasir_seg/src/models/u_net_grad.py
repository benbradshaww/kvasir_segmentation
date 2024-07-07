import tensorflow as tf

class CheckpointedConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, activation='relu', padding='same', **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size, activation=activation, padding=padding)

    @tf.recompute_grad
    def call(self, inputs):
        return self.conv(inputs)

class CheckpointedConv2DTranspose(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=2, activation='relu', padding='same', **kwargs):
        super().__init__(**kwargs)
        self.conv_transpose = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides, activation=activation, padding=padding)

    @tf.recompute_grad
    def call(self, inputs):
        return self.conv_transpose(inputs)

def build_unet(input_shape=(256, 256, 1)):
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder (downsampling)
    conv1 = CheckpointedConv2D(64, 3)(inputs)
    conv1 = CheckpointedConv2D(64, 3)(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = CheckpointedConv2D(128, 3)(pool1)
    conv2 = CheckpointedConv2D(128, 3)(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bridge
    conv3 = CheckpointedConv2D(256, 3)(pool2)
    conv3 = CheckpointedConv2D(256, 3)(conv3)

    # Decoder (upsampling)
    up4 = CheckpointedConv2DTranspose(128, 2)(conv3)
    up4 = tf.keras.layers.concatenate([up4, conv2])
    conv4 = CheckpointedConv2D(128, 3)(up4)
    conv4 = CheckpointedConv2D(128, 3)(conv4)

    up5 = CheckpointedConv2DTranspose(64, 2)(conv4)
    up5 = tf.keras.layers.concatenate([up5, conv1])
    conv5 = CheckpointedConv2D(64, 3)(up5)
    conv5 = CheckpointedConv2D(64, 3)(conv5)

    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(conv5)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Create the model
model = build_unet()

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Now you can train the model using model.fit()
# model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)