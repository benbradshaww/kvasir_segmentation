import tensorflow as tf;
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, MaxPooling2D, Conv2DTranspose, BatchNormalization, Concatenate

def conv_block(input, num_filters):
    '''
    Create a block of two convolutional layers with batch normalization and ReLU activation.

    Args:
        input (Tensor): Input tensor.
        num_filters (int): Number of filters for the convolutional layers.

    Returns:
        x (Tensor): Output tensor after applying two convolutional layers, batch normalization, and ReLU activation.
    '''

    # Convolutional Layer
    x = Conv2D(filters=num_filters,
               kernel_size=(3,3),
               padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=regularizers.L2(0.001),
               use_bias='False')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Convolutional Layer
    x = Conv2D(filters=num_filters,
               kernel_size=(3,3),
               padding='same', 
               kernel_initializer='he_normal',
               kernel_regularizer=regularizers.L2(0.001),
               use_bias='False')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x

def encoder_block(input, num_filters):
    '''
    Create a U-Net encoder block.

    Args:
        input (Tensor): Input tensor.
        num_filters (int): Number of filters for the convolutional layers.

    Returns:
        x (Tensor): Output tensor after the convolutional block.
        p (Tensor): Output tensor after max-pooling.
    '''

    # Convolutional Block
    x = conv_block(input=input, num_filters=num_filters)

    # Max Pooling
    p = MaxPooling2D(strides=(2,2),
                     pool_size=(2,2))(x)

    return x, p

    
def decoder_block(input, skip_features, num_filters):
    """
    Create a U-Net decoder block.

    Args:
        input (Tensor): Input tensor.
        skip_features (Tensor): Tensor from the encoder to be concatenated with the input.
        num_filters (int): Number of filters for the convolutional layers.

    Returns:
        x (Tensor): Output tensor after the decoder block.
    """

    # Transposed convolutional layer
    x = Conv2DTranspose(filters=num_filters, 
                        kernel_size=(2, 2),
                        strides=(2,2),
                        kernel_initializer='he_uniform',
                        kernel_regularizer=regularizers.l2(1e-5),
                        padding="same")(input)

    # Concatenation with skip connection features
    x = Concatenate()([x, skip_features])

    # Convolutional block
    x = conv_block(x, num_filters)

    return x


def u_net(input_shape):
    '''
    Build a U-Net model for image segmentation.

    Args:
        input_shape (tuple): Shape of the input image (height, width, channels).
        n_classes (int): Number of classes for segmentation.

    Returns:
        model (Model): U-Net model.
    '''

    inputs = Input(shape=input_shape)

    # Encoder blocks
    s1, p1 = encoder_block(input=inputs,
                           num_filters=64)
    s2, p2 = encoder_block(input=p1,
                           num_filters=128)
    s3, p3 = encoder_block(input=p2,
                           num_filters=256)
    s4, p4 = encoder_block(input=p3,
                           num_filters=512)

    # Bridge
    b1 = conv_block(p4, 1024)

    # Decoder blocks
    d1 = decoder_block(input=b1,
                       skip_features=s4,
                       num_filters=512)
    d2 = decoder_block(input=d1,
                       skip_features=s3,
                       num_filters=256)
    d3 = decoder_block(input=d2,
                       skip_features=s2,
                       num_filters=128)
    d4 = decoder_block(input=d3,
                       skip_features=s1,
                       num_filters=64)

    # 1x1 convolutions
    outputs = Conv2D(filters=1,
                     kernel_size=1,
                     padding="same",
                     activation='sigmoid')(d4) 

    model = Model(inputs=inputs,
                  outputs=outputs,
                  name='U-Net')

    return model
