import logging

import SimpleITK as sitk
import numpy as np
from skimage import measure


def undo_generator_steps(ndarray, cfg, interpol=sitk.sitkLinear, orig_sitk=None):
    """
    Undo the generator steps for a 3D volume
    1. calculate the intermediate size (which was used in the generator)
    2. Pad and crop to the intermediate size
    3. set the spacing given by the config
    4. Resample to the original spacing
    Parameters
    ----------
    ndarray :
    p :
    cfg :
    interpol :
    orig_sitk :

    Returns
    -------

    """
    from src.data.Preprocess import resample_3D, pad_and_crop
    from src.data.Preprocess import calc_resampled_size

    orig_size_ = orig_sitk.GetSize()
    orig_spacing_ = orig_sitk.GetSpacing()
    logging.debug('original shape: {}'.format(orig_size_))
    logging.debug('original spacing: {}'.format(orig_spacing_))

    # numpy has the following order: h,w,c (or z,h,w,c for 3D)
    w_h_size_sitk = orig_size_
    w_h_spacing_sitk = orig_spacing_

    # calculate the size of the image before crop or pad
    # z, x, y -- after reverse --> y,x,z we set this spacing to the input nda before resampling
    cfg_spacing = np.array((orig_spacing_[-1], *cfg['SPACING']))
    cfg_spacing = list(reversed(cfg_spacing))
    new_size = calc_resampled_size(orig_sitk, cfg_spacing)
    new_size = list(reversed(new_size))

    # pad, crop to original physical size in current spacing
    logging.debug('pred shape: {}'.format(ndarray.shape))
    logging.debug('intermediate size after pad/crop: {}'.format(new_size))

    ndarray = pad_and_crop(ndarray, new_size)
    logging.debug(ndarray.shape)

    # resample, set current spacing
    img_ = sitk.GetImageFromArray(ndarray)
    img_.SetSpacing(tuple(cfg_spacing))
    img_ = resample_3D(img_, size=w_h_size_sitk, spacing=w_h_spacing_sitk, interpolate=interpol)

    logging.debug('Size after resampling into original spacing: {}'.format(img_.GetSize()))
    logging.debug('Spacing after undo function: {}'.format(img_.GetSpacing()))

    return img_


def clean_3d_prediction_3d_cc(pred):
    """
    Find the biggest connected component per label
    This is a debugging method, which will plot each step
    returns: a tensor with the same shape as pred, but with only one cc per label
    """

    # avoid labeling images with float values
    assert len(np.unique(pred)) < 10, 'to many labels: {}'.format(len(np.unique(pred)))

    cleaned = np.zeros_like(pred)

    def clean_3d_label(val):

        """
        has access to pred, no passing required
        """

        # create a placeholder
        biggest = np.zeros_like(pred)
        biggest_size = 0

        # find all cc for this label
        # tensorflow operation is only in 2D
        # all_labels = tfa.image.connected_components(np.uint8(pred==val)).numpy()
        all_labels = measure.label(np.uint8(pred == val), background=0)

        for c in np.unique(all_labels)[1:]:
            mask = all_labels == c
            mask_size = mask.sum()
            if mask_size > biggest_size:
                biggest = mask
                biggest_size = mask_size
        return biggest

    for val in np.unique(pred)[1:]:
        biggest = clean_3d_label(val)
        cleaned[biggest] = val
    return cleaned


import cv2


def clean_3d_prediction_2d_cc(pred):
    cleaned = []
    # for each slice
    for s in pred:
        new_img = np.zeros_like(s)  # step 1
        # for each label
        for val in np.unique(s)[1:]:  # step 2
            mask = np.uint8(s == val)  # step 3
            labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]  # step 4
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # step 5
            new_img[labels == largest_label] = val  # step 6
        cleaned.append(new_img)
    return np.stack(cleaned, axis=0)
