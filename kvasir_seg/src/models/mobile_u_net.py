import tensorflow as tf
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import DepthwiseConv2D, BatchNormalization, ReLU, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, Input

def depthwise_separable_conv(inputs, num_filters):
    '''
    Create a block of two depthwise seperable convolutional layers with batch normalization and ReLU activation.

    Args:
        input (Tensor): Input tensor.
        num_filters (int): Number of filters for the convolutional layers.

    Returns:
        x (Tensor): Output tensor after applying two convolutional layers, batch normalization, and ReLU activation.
    '''
    # Depthwise Seperable Convolution Layer
    x = DepthwiseConv2D(kernel_size = (3,3),
                        padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.L2(0.001),
                        use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=num_filters,
               kernel_size=(1, 1),
               padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=regularizers.L2(0.001),
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Depthwise Seperable Convolution Layer
    x = DepthwiseConv2D(kernel_size = (3,3),
                        padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.L2(0.001),
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=num_filters,
               kernel_size=(1, 1),
               padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=regularizers.L2(0.001),
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x

def encoder_block(inputs, num_filters):
    '''
    Create a Mobile-U-Net encoder block.

    Args:
        input (Tensor): Input tensor.
        num_filters (int): Number of filters for the convolutional layers.

    Returns:
        x (Tensor): Output tensor after the convolutional block.
        p (Tensor): Output tensor after max-pooling.
    '''
    # Depthwise Seperable Convolution Block
    x = depthwise_separable_conv(inputs, num_filters)

    # Max Pooling
    p = MaxPooling2D(2)(x)

    return x, p

def decoder_block(inputs, skip_features, num_filters):
    """
    Create a Mobile-U-Net decoder block.

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
                        padding="same")(inputs)
    # Concatenation with skip connection features
    x = Concatenate()([x, skip_features])

    # Seperable Depthwise Convolution Block
    x = depthwise_separable_conv(x, num_filters)

    return x

def mobile_u_net(input_shape):
    '''
    Build a Mobile-U-Net model for image segmentation.

    Args:
        input_shape (tuple): Shape of the input image (height, width, channels).
        n_classes (int): Number of classes for segmentation.

    Returns:
        model (Model): U-Net model.
    '''
    inputs = Input(shape=input_shape)
    
    # Encoder
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)
    
    # Bridge
    b1 = depthwise_separable_conv(p4, 1024)
    
    # Decoder
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    
    # Output
    # 1x1 convolutions
    outputs = tf.cast(Conv2D(filters=1,
                             kernel_size=1,
                             padding="same",
                             activation='sigmoid')(d4), dtype='float32')
    
    model = Model(inputs=inputs, outputs=outputs, name='Mobile-U-Net')

    return model
