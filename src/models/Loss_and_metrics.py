from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from functools import partial
from tensorflow.keras.losses import mse
from src.data.Dataset import get_metadata_maybe, ensure_dir

def max_volume_loss(min_probability=0.8,):
    """
    Create a callable loss function which maximizes the probability values of y_pred
    There is additionally the possibility to weight high probabilities in
    :param min_probability:
    :return: loss function which maximize the number of voxels with a probability higher than the threshold
    """

    def max_loss(y_true, y_pred):
        """
        Maximize the foreground voxels in the middle slices with a probability higher than a given threshold.
        :param y_true:
        :param y_pred:
        :param weights:
        :return:
        """

        # ignore background channel if given, we want to maximize the number of captured foreground voxel
        if y_pred.shape[-1] == 4:
            y_pred = y_pred[...,1:]
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        sum_bigger_than = tf.reduce_max(y_pred, axis=-1)
        mask_bigger_than = tf.cast(sum_bigger_than > min_probability, tf.float32)
        sum_bigger_than = sum_bigger_than * mask_bigger_than

        return 1- tf.reduce_mean(sum_bigger_than)

    return max_loss


def loss_with_zero_mask(loss=mse, mask_smaller_than=0.01, weight_inplane=False,xy_shape=224):
    """
    Loss-factory returns a loss which calculates a given loss-function (e.g. MSE) only for the region where y_true is greater than a given threshold
    This is necessary for our AX2SAX comparison, as we have different length of CMR stacks (AX2SAX gt is cropped at z = SAX.z + 20mm)
    Example in-plane weighting which is multiplied to each slice of the volume
    [[0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ]
     [0.   0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.  ]
     [0.   0.25 0.5  0.5  0.5  0.5  0.5  0.5  0.25 0.  ]
     [0.   0.25 0.5  0.75 0.75 0.75 0.75 0.5  0.25 0.  ]
     [0.   0.25 0.5  0.75 1.   1.   0.75 0.5  0.25 0.  ]
     [0.   0.25 0.5  0.75 1.   1.   0.75 0.5  0.25 0.  ]
     [0.   0.25 0.5  0.75 0.75 0.75 0.75 0.5  0.25 0.  ]
     [0.   0.25 0.5  0.5  0.5  0.5  0.5  0.5  0.25 0.  ]
     [0.   0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.  ]
     [0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ]]
    :param loss: any callable loss function. e.g. tf.keras.losses
    :param mask_smaller_than: float, threshold to calculate the loss only for voxels where gt is greater
    :param weight_inplane: bool, apply in-plane weighting
    :param xy_shape: int, number of square in-plane pixels
    :return:
    """

    # in-plane weighting, which helps to focus on the voxels close to the center
    x_shape = xy_shape
    y_shape = xy_shape
    temp = np.zeros((x_shape, y_shape))
    weights_distribution = np.linspace(0, 100, x_shape // 2)
    for i, l in enumerate(weights_distribution):
        temp[i:-i, i:-i] = l
    weights = temp[None, None, :, :]
    weights = tf.convert_to_tensor(weights, dtype=tf.float32)

    def my_loss(y_true, y_pred, weights_inplane=weights):
        """
        wrapper to either calculate a loss only on areas where the gt is greater than mask_smaller_than
        and additionally weight the loss in-plane to increase the importance of the voxels close to the center
        :param y_true:
        :param y_pred:
        :return:
        """
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        y_true = tf.cast(y_true, dtype=tf.float32)
        mask = tf.squeeze(tf.cast((y_true > mask_smaller_than),tf.float32),axis=-1)

        if weight_inplane:
            return (loss(y_true, y_pred) * mask) * weights_inplane + K.epsilon()
        else:
            return loss(y_true, y_pred) * mask

    return my_loss


# modified with dice coef applied
# a weighted cross entropy loss function combined with the dice coef for faster learning
def weighted_cce_dice_coef(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probs of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    def cat_cross_entropy_dice_coef(y_true, y_pred):
        return loss(y_true, y_pred)- dice_coef(y_true, y_pred)
    
    return cat_cross_entropy_dice_coef

def dice_coef_background(y_true, y_pred):
    y_pred = y_pred[...,0]
    y_true = y_true[...,0]
    return dice_coef(y_true, y_pred)

def dice_coef_rv(y_true, y_pred):
    y_pred = y_pred[...,-3]
    y_true = y_true[...,-3]
    return dice_coef(y_true, y_pred)

def dice_coef_lower(y_true, y_pred): #LA_changed from myo
    y_pred = y_pred[...,-2]
    y_true = y_true[...,-2]
    return dice_coef(y_true, y_pred)

def dice_coef_upper(y_true, y_pred): ##LA_changed from LV
    y_pred = y_pred[...,-1]
    y_true = y_true[...,-1]
    return dice_coef(y_true, y_pred)

def dice_coef_myo(y_true, y_pred): #LA_changed from myo
    y_pred = y_pred[...,-2]
    y_true = y_true[...,-2]
    return dice_coef(y_true, y_pred)

def dice_coef_lv(y_true, y_pred): ##LA_changed from LV
    y_pred = y_pred[...,-1]
    y_true = y_true[...,-1]
    return dice_coef(y_true, y_pred)


# ignore background score
# LA: combination of all channels that are being used, not a seperate third metric. 
def dice_coef_labels(y_true, y_pred): 

    # ignore background, slice from the back to work with and without background channels
    y_pred = y_pred[...,-3:]
    y_true = y_true[...,-3:]
    
    return dice_coef(y_true, y_pred)

def dice_coef(y_true, y_pred):
    smooth = 1.
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_squared(y_true, y_pred):
    smooth = 1.
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true_f)) + K.sum(K.square(y_pred_f)) + smooth)


def dice_numpy(y_true, y_pred, empty_score=1.0):

    """
    Hard Dice for numpy ndarrays
    :param y_true:
    :param y_pred:
    :param empty_score:
    :return:
    """

    im1 = np.asarray(y_true).astype(np.bool)
    im2 = np.asarray(y_pred).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum

class BceDiceLoss(tf.keras.losses.Loss):

    def __init__(self, w_bce=1., w_dice=1., binary=True, name='BcdDiceLoss'):

        super().__init__(name='{}_w_{}_{}'.format(name,w_bce, w_dice))
        self.w_bce = w_bce
        self.w_dice = w_dice
        if binary:
            self.entropy = tf.keras.losses.binary_crossentropy
        else:
            self. entropy = tf.keras.losses.categorical_crossentropy

    def __call__(self, y_true, y_pred, **kwargs):
        # use only the foreground labels for the loss
        if y_pred.shape[-1] == 4:
            y_pred = y_pred[..., -3:]
            y_true = y_true[..., -3:]

        return (self.entropy(y_true, y_pred)* self.w_bce) - (dice_coef(y_true, y_pred)*self.w_dice)


def bce_dice_loss(y_true, y_pred, w_bce=0.5, w_dice=1.):
    """
    weighted binary cross entropy - dice coef loss
    uses all labels if shape labels == 3
    otherwise slice the background to ignore over-represented background class
    :param y_true:
    :param y_pred:
    :return:
    """

    # use only the labels for the loss
    if y_pred.shape[-1] == 4:
        y_pred = y_pred[...,-3:]
        y_true = y_true[...,-3:]


    return w_bce * tf.keras.losses.binary_crossentropy(y_true, y_pred) - w_dice * dice_coef(y_true, y_pred)


# experimental, does not work
# def cce_dice_loss(y_true, y_pred, w_cce=0.5, w_dice=1.):
#     """
#     weighted binary cross entropy - dice coef loss
#     uses all labels if shape labels == 3
#     otherwise slice the background to ignore over-represented background class
#     :param y_true:
#     :param y_pred:
#     :return:
#     """

#     # use only the labels for the loss
#     if y_pred.shape[-1] == 4:
#         y_pred = y_pred[...,-3:]
#         y_true = y_true[...,-3:]


#     return w_cce * tf.keras.losses.CategoricalCrossentropy(y_true, y_pred) - w_dice * dice_coef(y_true, y_pred)
