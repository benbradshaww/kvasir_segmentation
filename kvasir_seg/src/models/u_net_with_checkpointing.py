import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, MaxPooling2D, Conv2DTranspose, BatchNormalization, Concatenate

class CheckpointedLayer(tf.keras.layers.Layer):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def build(self, input_shape):
        self.layer.build(input_shape)
        self._trainable_weights = self.layer.trainable_weights
        self._non_trainable_weights = self.layer.non_trainable_weights

    @tf.custom_gradient
    def call(self, x):
        def grad(dy, variables=None):
            with tf.GradientTape() as tape:
                tape.watch(x)
                y = self.layer(x)
            grads = tape.gradient(y, [x] + self.layer.trainable_weights, dy)
            return grads[0], grads[1:]
        return self.layer(x), grad

def checkpointed_conv_block(input, num_filters):
    x = CheckpointedLayer(Conv2D(filters=num_filters,
                                 kernel_size=(3,3),
                                 padding='same',
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=regularizers.L2(0.001),
                                 use_bias=False))(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = CheckpointedLayer(Conv2D(filters=num_filters,
                                 kernel_size=(3,3),
                                 padding='same', 
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=regularizers.L2(0.001),
                                 use_bias=False))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x

def checkpointed_encoder_block(input, num_filters):
    x = checkpointed_conv_block(input=input, num_filters=num_filters)
    p = MaxPooling2D(strides=(2,2), pool_size=(2,2))(x)
    return x, p

def checkpointed_decoder_block(input, skip_features, num_filters):
    x = CheckpointedLayer(Conv2DTranspose(filters=num_filters, 
                                          kernel_size=(2, 2),
                                          strides=(2,2),
                                          kernel_initializer='he_uniform',
                                          kernel_regularizer=regularizers.l2(1e-5),
                                          padding="same"))(input)
    x = Concatenate()([x, skip_features])
    x = checkpointed_conv_block(x, num_filters)
    return x

def u_net_with_checkpointing(input_shape):
    inputs = Input(shape=input_shape)

    # Encoder blocks
    s1, p1 = checkpointed_encoder_block(input=inputs, num_filters=64)
    s2, p2 = checkpointed_encoder_block(input=p1, num_filters=128)
    s3, p3 = checkpointed_encoder_block(input=p2, num_filters=256)
    s4, p4 = checkpointed_encoder_block(input=p3, num_filters=512)

    # Bridge
    b1 = checkpointed_conv_block(p4, 1024)

    # Decoder blocks
    d1 = checkpointed_decoder_block(input=b1, skip_features=s4, num_filters=512)
    d2 = checkpointed_decoder_block(input=d1, skip_features=s3, num_filters=256)
    d3 = checkpointed_decoder_block(input=d2, skip_features=s2, num_filters=128)
    d4 = checkpointed_decoder_block(input=d3, skip_features=s1, num_filters=64)

    # Output layer
    outputs = CheckpointedLayer(Conv2D(filters=1,
                                       kernel_size=1,
                                       padding="same",
                                       activation='sigmoid'))(d4)

    model = Model(inputs=inputs, outputs=outputs, name='U-Net-Checkpointed')

    return model