import logging
import os
import sys

import SimpleITK as sitk
import cv2
import numpy as np
import skimage
from albumentations import GridDistortion, RandomRotate90, ReplayCompose, Downscale, ShiftScaleRotate
from sklearn.preprocessing import RobustScaler

from src.data.Dataset import get_metadata_maybe


def get_ip_from_2dmask(nda, debug=False, rev=False):
    """
    Find the RVIP on a 2D mask with the following labels
    RV (0), LVMYO (1) and LV (2) mask

    Parameters
    ----------
    nda : numpy ndarray with one hot encoded labels
    debug :

    Returns a tuple of two points anterior IP, inferior IP, each with (y,x)-coordinates
    -------

    """
    if debug: print('msk shape: {}'.format(nda.shape))
    # initialise some values
    first, second = None, None
    # find first and second insertion points
    myo_msk = (nda == 2).astype(np.uint8)
    comb_msk = ((nda == 1) | (nda == 2) | (nda == 3)).astype(np.uint8)
    if np.isin(1, nda) and np.isin(2, nda):
        myo_contours, hierarchy = cv2.findContours(myo_msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        comb_contours, hierarchy = cv2.findContours(comb_msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(myo_contours) > 0 and len(comb_contours) > 0:  # we just need to search for IP if there are two contours
            # some lambda helpers
            # transform and describe contour lists to pythonic list which makes "elem in" syntax possible
            clean_contour = lambda cont: list(map(lambda x: (x[0][0], x[0][1]), cont[0]))
            descr_cont = lambda cont: print(
                'len: {}, first elem: {}, type of one elem: {}'.format(len(cont), cont[0], type(cont[0])))

            # clean/describe both contours
            myo_clean = clean_contour(myo_contours)
            if debug: descr_cont(myo_clean)
            comb_clean = clean_contour(comb_contours)
            if debug: descr_cont(comb_clean)

            # initialise some values
            septum_visited = False
            border_visited = False
            memory_first = None
            for p in myo_clean:
                if debug: print('p {} in {}'.format(p, p in comb_clean))
                # we are at the border,
                # moving anti-clockwise,
                # we dont know if we are in the septum
                # no second IP found so far.

                if p in comb_clean:
                    border_visited = True
                    if septum_visited and not second:
                        # take the first point after the septum as second IP
                        # we are at the border
                        # we have been at the septum
                        # no second defined so far
                        second = p
                        if debug: print('second= {}'.format(second))

                    # we are at the border
                    if not first:
                        # if we haven't been at the septum, update/remember this point
                        # use the last visited point before visiting the septum as first IP
                        memory_first = p
                        if debug: print('memory= {}'.format(memory_first))
                else:
                    septum_visited = True  # no contour points matched --> we are at the septum
                    if border_visited and not first:
                        first = memory_first
            if second and not first:  # if our contour started at the first IP
                first = memory_first
            # assert first and second, 'missed one insertion point: first: {}, second: {}'.format(first, second)
            if debug: print('first IP: {}, second IP: {}'.format(first, second))
        if rev and (first is not None) and (second is not None): first, second = (first[1], first[0]), (
            second[1], second[0])

    return first, second


def get_ip_from_mask_3d(msk_3d, debug=False, keepdim=False, rev=False):
    '''
    Returns two lists of RV insertion points (y,x)-coordinates
    For a standard SAX orientation:
    the first list belongs to the anterior IP and the second to the inferior IP
    Parameters
    ----------
    msk_3d : (np.ndarray) with z,y,x
    debug : (bool) print additional info
    keepdim: (bool) returns two lists of the same length as z, slices where no RV IPs were found are represented by an tuple of None
    rev: (bool), return the coordinates as x,y tuples (for comparison with matrix based indexing)

    Returns tuple of lists with points (y,x)-coordinates
    -------

    '''
    first_ips = []
    second_ips = []
    for msk2d in msk_3d:
        try:
            first, second = get_ip_from_2dmask(msk2d, debug=debug, rev=rev)
            if (first is not None) and (second is not None) or keepdim:
                first_ips.append(first)
                second_ips.append(second)
        except Exception as e:
            print(str(e))
            pass

    return first_ips, second_ips


def calc_resampled_size(sitk_img, target_spacing):
    # calculate the new size of each 3D volume (after resample with the given spacing)
    # sitk.spacing has the opposite order than np.shape and tf.shape
    # In the config we use the numpy order z, y, x which needs to be reversed for sitk
    if type(target_spacing) in [list, tuple]:
        target_spacing = np.array(target_spacing)
    old_size = np.array(sitk_img.GetSize())
    old_spacing = np.array(sitk_img.GetSpacing())
    logging.debug('old size: {}, old spacing: {}, target spacing: {}'.format(old_size, old_spacing,
                                                                             target_spacing))
    new_size = (old_size * old_spacing) / target_spacing
    return list(np.around(new_size).astype(np.int))


def load_masked_img(sitk_img_f, mask=False, masking_values=[1, 2, 3], replace=('img', 'msk'), mask_labels=[0, 1, 2, 3]):
    """
    Wrapper for opening a dicom image, this wrapper could also load the corresponding segmentation map and mask the loaded image on the fly
     if mask == True use the replace wildcard to open the corresponding segmentation mask
     Use the values given in mask_labels to transform the one-hot-encoded mask into channel based binary mask
     Mask/cut the CMR image/volume by the given labels in masking_values

    Parameters
    ----------
    sitk_img_f : full filename for a dicom image/volume, could be any format supported by sitk
    mask : bool, if the sitk image loaded should be cropped by any label of the corresponding mask
    masking_values : list of int, defines the area/labels which should be cropped from the original CMR
    replace : tuple of replacement string to get from the image filename to the mask filename
    mask_labels : list of int
    """

    assert os.path.isfile(sitk_img_f), 'no valid image: {}'.format(sitk_img_f)
    img_original = sitk.ReadImage(sitk_img_f, sitk.sitkFloat32)

    if mask:
        sitk_mask_f = sitk_img_f.replace(replace[0], replace[1])
        msk_original = sitk.ReadImage(sitk_mask_f)

        img_nda = sitk.GetArrayFromImage(img_original)
        msk_nda = transform_to_binary_mask(sitk.GetArrayFromImage(msk_original), mask_values=mask_labels)

        # mask by different labels, sum up all masked channels
        temp = np.zeros(img_nda.shape)
        for c in masking_values:
            # mask by different labels, sum up all masked channels
            temp += img_nda * msk_nda[..., c].astype(np.bool)
        sitk_img = sitk.GetImageFromArray(temp)

        # copy metadata
        for tag in img_original.GetMetaDataKeys():
            value = get_metadata_maybe(img_original, tag)
            sitk_img.SetMetaData(tag, value)
        sitk_img.SetSpacing(img_original.GetSpacing())
        sitk_img.SetOrigin(img_original.GetOrigin())

        img_original = sitk_img

    return img_original


def resample_3D(sitk_img, size=(256, 256, 12), spacing=(1.25, 1.25, 8), interpolate=sitk.sitkNearestNeighbor):
    """
    resamples an 3D sitk image or numpy ndarray to a new size with respect to the giving spacing
    This method expects size and spacing in sitk format: x, y, z
    :param sitk_img:
    :param size:
    :param spacing:
    :param interpolate:
    :return: returns the same datatype as submitted, either sitk.image or numpy.ndarray
    """

    return_sitk = True

    if isinstance(sitk_img, np.ndarray):
        return_sitk = False
        sitk_img = sitk.GetImageFromArray(sitk_img)

    assert (isinstance(sitk_img, sitk.Image)), 'wrong image type: {}'.format(type(sitk_img))

    # make sure to have the correct data types
    size = [int(elem) for elem in size]
    spacing = [float(elem) for elem in spacing]

    # if len(size) == 3 and size[0] < size[-1]: # 3D data, but numpy shape and size, reverse order for sitk
    # bug if z is lonnger than x or y
    #    size = tuple(reversed(size))
    #    spacing = tuple(reversed(spacing))
    # logging.error('spacing in resample 3D: {}'.format(sitk_img.GetSpacing()))
    # logging.error('size in resample 3D: {}'.format(sitk_img.GetSize()))
    # logging.error('target spacing in resample 3D: {}'.format(spacing))
    # logging.error('target size in resample 3D: {}'.format(size))

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolate)
    resampler.SetSize(size)
    resampler.SetOutputDirection(sitk_img.GetDirection())
    resampler.SetOutputSpacing(spacing)
    resampler.SetOutputOrigin(sitk_img.GetOrigin())

    resampled = resampler.Execute(sitk_img)

    # return the same data type as input datatype
    if return_sitk:
        return resampled
    else:
        return sitk.GetArrayFromImage(resampled)


def augmentation_compose_2d_3d_4d(img, mask, probabillity=1, config=None):
    """
    Apply an compisition of different augmentation steps,
    either on 2D or 3D image/mask pairs,
    apply
    :param img:
    :param mask:
    :param probabillity:
    :return: augmented image, mask
    """
    # logging.debug('random rotate for: {}'.format(img.shape))
    return_image_and_mask = True
    img_given = True
    mask_given = True

    if isinstance(img, sitk.Image):
        img = sitk.GetArrayFromImage(img).astype(np.float32)

    if isinstance(mask, sitk.Image):
        mask = sitk.GetArrayFromImage(mask).astype(np.float32)

    # dont print anything if no images nor masks are given
    if img is None and mask is None:
        logging.error('No image data given')
        raise ('No image data given in augmentation compose')

    # replace mask with empty slice if none is given
    if mask is None:
        return_image_and_mask = False
        mask_given = False

    # replace image with empty slice if none is given
    if img is None:
        return_image_and_mask = False
        img_given = False

    targets = {}
    data = {}
    img_placeholder = 'image'
    mask_placeholder = 'mask'

    if img.ndim == 2:
        data = {"image": img, "mask": mask}

    if img.ndim == 3:
        middle_z = len(img) // 2
        if mask_given:
            m_ = mask[middle_z]
        else:
            m_ = mask
        # take an image, mask pair from the middle part of the volume
        data = {"image": img[middle_z], "mask": m_}

        # add each slice of the image/mask stacks into the data dictionary
        for z in range(img.shape[0]):
            # add the other slices to the data dict
            if img_given: data['{}{}'.format(img_placeholder, z)] = img[z, ...]
            if mask_given: data['{}{}'.format(mask_placeholder, z)] = mask[z, ...]
            # define the target group,
            # which slice is a mask and which an image (different interpolation)
            if img_given: targets['{}{}'.format(img_placeholder, z)] = 'image'
            if mask_given: targets['{}{}'.format(mask_placeholder, z)] = 'mask'

    if img.ndim == 4:
        middle_t = img.shape[0] // 2
        middle_z = img.shape[1] // 2
        # take an image, mask pair from the middle part of the volume and time
        if mask_given:
            data = {"image": img[middle_t][middle_z], "mask": m_}
        else:
            data = {"image": img[middle_t][middle_z]}

        for t in range(img.shape[0]):
            # add each slice of the image/mask stacks into the data dictionary
            for z in range(img.shape[1]):
                # add the other slices to the data dict
                if img_given: data['{}_{}_{}'.format(img_placeholder, t, z)] = img[t, z, ...]
                if mask_given: data['{}_{}_{}'.format(mask_placeholder, t, z)] = mask[t, z, ...]
                # define the target group,
                # which slice is a mask and which an image (different interpolation)
                if img_given: targets['{}_{}_{}'.format(img_placeholder, t, z)] = 'image'
                if mask_given: targets['{}_{}{}'.format(mask_placeholder, t, z)] = 'mask'

    # create a callable augmentation composition
    aug = _create_aug_compose(p=probabillity, targets=targets, config=config)

    # apply the augmentation
    augmented = aug(**data)
    logging.debug(augmented['replay'])

    if img.ndim == 3:
        images = []
        masks = []
        for z in range(img.shape[0]):
            # extract the augmented slices in the correct order
            if img_given: images.append(augmented['{}{}'.format(img_placeholder, z)])
            if mask_given: masks.append(augmented['{}{}'.format(mask_placeholder, z)])
        if img_given: augmented['image'] = np.stack(images, axis=0)
        if mask_given: augmented['mask'] = np.stack(masks, axis=0)

    if img.ndim == 4:
        img_4d = []
        mask_4d = []
        for t in range(img.shape[0]):
            images = []
            masks = []
            for z in range(img.shape[1]):
                # extract the augmented slices in the correct order
                if img_given: images.append(augmented['{}_{}_{}'.format(img_placeholder, t, z)])
                if mask_given: masks.append(augmented['{}_{}_{}'.format(mask_placeholder, t, z)])
            if img_given: img_4d.append(np.stack(images, axis=0))
            if mask_given: mask_4d.append(np.stack(masks, axis=0))

        if img_given: augmented['image'] = np.stack(img_4d, axis=0)
        if mask_given: augmented['mask'] = np.stack(mask_4d, axis=0)

    if return_image_and_mask:
        return augmented['image'], augmented['mask']
    else:
        # dont return the fake augmented masks if none where given
        return augmented['image']


def match_2d_on_nd(nda, avg):
    if nda.ndim == 2:
        return match_2d_hist_on_2d(nda, avg)
    elif nda.ndim == 3:
        return match_2d_hist_on_3d(nda, avg)
    elif nda.ndim == 4:
        return match_2d_hist_on_4d(nda, avg)
    else:
        logging.info('shape for histogram matching does not fit to any method, return unmodified nda')
        return nda


def match_2d_hist_on_2d(nda, avg):
    return skimage.exposure.match_histograms(nda, avg, multichannel=False)


def match_2d_hist_on_3d(nda, avg):
    for z in range(nda.shape[0]):
        nda[z] = skimage.exposure.match_histograms(nda[z], avg, multichannel=False)
    return nda


def match_2d_hist_on_4d(nda, avg):
    for t in range(nda.shape[0]):
        for z in range(nda.shape[1]):
            nda[t, z] = skimage.exposure.match_histograms(nda[t, z], avg, multichannel=False)
    return nda


def _create_aug_compose(p=1, border_mode=cv2.BORDER_CONSTANT, val=0, targets=None, config=None):
    """
    Create an Albumentations Reply compose augmentation based on the config params
    Parameters
    ----------
    p :
    border_mode :
    val :
    targets :
    config :
    Note for the border mode from openCV:
    BORDER_CONSTANT    = 0,
    BORDER_REPLICATE   = 1,
    BORDER_REFLECT     = 2,
    BORDER_WRAP        = 3,
    BORDER_REFLECT_101 = 4,
    BORDER_TRANSPARENT = 5,
    BORDER_REFLECT101  = BORDER_REFLECT_101,
    BORDER_DEFAULT     = BORDER_REFLECT_101,
    BORDER_ISOLATED    = 16,

    Returns
    -------

    """
    if config is None:
        config = {}
    if targets is None:
        targets = {}
    prob = config.get('AUGMENT_PROB', 0.8)
    border_mode = config.get('BORDER_MODE', border_mode)
    val = config.get('BORDER_VALUE', val)
    augmentations = []
    if config.get('RANDOMROTATE', False): augmentations.append(RandomRotate90(p=0.2))
    if config.get('SHIFTSCALEROTATE', False): augmentations.append(
        ShiftScaleRotate(p=prob, rotate_limit=0, shift_limit=0.025, scale_limit=0, value=val, border_mode=border_mode))
    if config.get('GRIDDISTORTION', False): augmentations.append(
        GridDistortion(p=prob, value=val, border_mode=border_mode))
    if config.get('DOWNSCALE', False): augmentations.append(Downscale(scale_min=0.9, scale_max=0.9, p=prob))
    return ReplayCompose(augmentations, p=p,
                         additional_targets=targets)


def transform_to_binary_mask(mask_nda, mask_values=[0, 1, 2, 3]):
    """
    Transform from a value-based representation to a binary channel based representation
    :param mask_nda:
    :param mask_values:
    :return:
    """
    # transform the labels to binary channel masks

    mask = np.zeros((*mask_nda.shape, len(mask_values)), dtype=np.bool)
    for ix, mask_value in enumerate(mask_values):
        mask[..., ix] = mask_nda == mask_value
    return mask


def from_channel_to_flat(binary_mask, start_c=0):
    """
    Transform a tensor or numpy nda from a channel-wise (one channel per label) representation
    to a value-based representation
    :param binary_mask:
    :return:
    """
    # convert to bool nda to allow later indexing
    binary_mask = binary_mask >= 0.5

    # reduce the shape by the channels
    temp = np.zeros(binary_mask.shape[:-1], dtype=np.uint8)

    for c in range(binary_mask.shape[-1]):
        temp[binary_mask[..., c]] = c + start_c
    return temp


def clip_quantile(img_nda, upper_quantile=.999, lower_boundary=0):
    """
    clip to values between 0 and .999 quantile
    :param img_nda:
    :param upper_quantile:
    :return:
    """

    ninenine_q = np.quantile(img_nda.flatten(), upper_quantile, overwrite_input=False)

    return np.clip(img_nda, lower_boundary, ninenine_q)


def normalise_image(img_nda, normaliser='minmax'):
    """
    Normalise Images to a given range,
    normaliser string repr for scaler, possible values: 'MinMax', 'Standard' and 'Robust'
    if no normalising method is defined use MinMax normalising
    :param img_nda:
    :param normaliser:
    :return:
    """
    # ignore case
    normaliser = normaliser.lower()

    if normaliser == 'standard':
        return (img_nda - np.mean(img_nda)) / (np.std(img_nda) + sys.float_info.epsilon)

        # return StandardScaler(copy=False, with_mean=True, with_std=True).fit_transform(img_nda)
    elif normaliser == 'robust':
        return RobustScaler(copy=False, quantile_range=(0.0, 95.0), with_centering=True,
                            with_scaling=True).fit_transform(img_nda)
    else:
        return (img_nda - img_nda.min()) / (img_nda.max() - img_nda.min() + sys.float_info.epsilon)


def pad_and_crop(ndarray, target_shape=(10, 10, 10)):
    """
    Center pad and crop a np.ndarray with any shape to a given target shape
    Parameters
    Pad and crop must be the complementary
    pad = floor(x),floor(x)+1
    crop = floor(x)+1, floor(x)
    ----------
    ndarray : numpy.ndarray of any shape
    target_shape : must have the same length as ndarray.ndim

    Returns np.ndarray with each axis either pad or crop
    -------

    """
    cropped = np.zeros(target_shape)
    target_shape = np.array(target_shape)
    logging.debug('input shape, crop_and_pad: {}'.format(ndarray.shape))
    logging.debug('target shape, crop_and_pad: {}'.format(target_shape))

    diff = ndarray.shape - target_shape

    # divide into summands to work with odd numbers
    # take the same numbers for left or right padding/cropping if the difference is dividable by 2
    # else take floor(x),floor(x)+1 for PAD (diff<0)
    # else take floor(x)+1, floor(x) for CROP (diff>0)
    d = list(
        (int(x // 2), int(x // 2)) if x % 2 == 0 else (int(np.floor(x / 2)), int(np.floor(x / 2) + 1)) if x < 0 else (
            int(np.floor(x / 2) + 1), int(np.floor(x / 2))) for x in diff)
    # replace the second slice parameter if it is None, which slice until end of ndarray
    d = list((abs(x), abs(y)) if y != 0 else (abs(x), None) for x, y in d)
    # create a bool list, negative numbers --> pad, else --> crop
    pad_bool = diff < 0
    crop_bool = diff > 0

    # create one slice obj for cropping and one for padding
    pad = list(i if b else (None, None) for i, b in zip(d, pad_bool))
    crop = list(i if b else (None, None) for i, b in zip(d, crop_bool))

    # Create one tuple of slice calls per pad/crop
    # crop or pad from dif:-dif if second param not None, else replace by None to slice until the end
    # slice params: slice(start,end,steps)
    pad = tuple(slice(i[0], -i[1]) if i[1] != None else slice(i[0], i[1]) for i in pad)
    crop = tuple(slice(i[0], -i[1]) if i[1] != None else slice(i[0], i[1]) for i in crop)

    # crop and pad in one step
    cropped[pad] = ndarray[crop]
    return cropped
