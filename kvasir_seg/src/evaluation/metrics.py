import tensorflow as tf

# Define Dice coefficient for evaluating segmentation performance
def dice_coefficient(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice

# Define IoU (Intersection over Union) coefficient
def iou_coefficient(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

# Define Dice loss based on Dice coefficient
def dice_loss(y_true, y_pred):
    loss = 1 - dice_coefficient(y_true, y_pred)
    return loss

# Define IoU loss based on IoU coefficient
def iou_loss(y_true, y_pred):
    loss = 1 - iou_coefficient(y_true, y_pred)
    return loss
