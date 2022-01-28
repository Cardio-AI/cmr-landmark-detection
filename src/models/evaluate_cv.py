# import external libs
import os, glob
import cv2
import pandas as pd
import numpy as np
import SimpleITK as sitk

from src.data.Preprocess import get_ip_from_mask_3d

def get_ip_from_rvip_file(f_name, keepdim=False):
    nda = sitk.GetArrayFromImage(sitk.ReadImage(f_name))
    return get_ip_from_rvip_mask_3d(nda, keepdim=keepdim)

def get_ip_from_ventriclemsk_file(f_name, keepdim=False, yx_coordinates=True):
    nda = sitk.GetArrayFromImage(sitk.ReadImage(f_name))
    return get_ip_from_mask_3d(nda, keepdim=keepdim, rev=yx_coordinates)

def get_ip_from_rvip_mask_3d(msk_3d, debug=False, keepdim=False):
    '''
    Returns two lists of RV insertion points (y,x)-coordinates
    For a standard SAX orientation:
    the first list belongs to the anterior IP and the second to the inferior IP
    Parameters
    ----------
    msk_3d : (np.ndarray) with z,y,x
    debug : (bool) print additional info
    keepdim: (bool) returns two lists of the same length as z, slices where no RV IPs were found are represented by an tuple of None

    Returns tuple of lists with points (y,x)-coordinates
    -------

    '''
    first_ips = []
    second_ips = []
    for msk2d in msk_3d:
        try:
            first, second = get_mean_rvip_2d(msk2d)
            if (first is not None and second is not None) or keepdim:  # note the object comparison 'is' instead of '='
                first_ips.append(first)
                second_ips.append(second)
        except Exception as e:
            print(str(e))
            pass

    return first_ips, second_ips


def get_mean_rvip_2d(nda_2d):
    """
    Expects a numpy ndarray/ mask with at least two one-hot encoded labels,
    pixels==1 represents the anterior IP, pixels==2 the inferior IP
    returns a list with len== number of unique values in nda,
    each element is a point (y/x-coordinates) (from an image perspective) pointing the mean coordinates of this value
    usually this will return [anterior IP, inferior IP],
    could also be used for more points
    """

    assert len(nda_2d.shape) == 2, 'invalid shape: {}'.format(nda_2d.shape)
    mean_points = []
    labels = np.unique(nda_2d)[1:]  # ignore background
    anterior, inferior = None, None  # return None for this slice, if no ips are available
    if len(labels) == 2:
        anterior, inferior = list(map(lambda x: np.where(nda_2d == x), labels))
        anterior = list(np.array(anterior).mean(axis=1))
        inferior = list(np.array(inferior).mean(axis=1))
    return anterior, inferior


def isvalid(number):
    if number is None or all(np.isnan(number)):
        return False
    else:
        return True


from math import atan2, degrees

def get_angles2x(rvips):
    ants, infs = rvips
    angles = np.array([get_angle2x(a, b) if (a is not None and b is not None) else None for a, b in zip(ants, infs)])
    return angles

def get_angle2x(p1, p2):
    '''
    Calc the angle between two points with a defined order
    p1==anterior, p2==inferior
    (anterior and inferior IPS) and the x-axis
    we expect point as y,x tuple
    Here we calculate always the angle between the x-axis anti-clock-wise and the ip-connection line
    Parameters
    ----------
    p1 : tuple y,x
    p2 : tuple y,x

    Returns angle in degree, None if at least one of the points are None
    -------

    '''
    try:
        angle = None
        if (np.isfinite(p1).all() and np.isfinite(p2).all()):
            y1, x1, y2, x2 = p1[0], p1[1], p2[0], p2[1]
            angle = degrees(atan2(y2 - y1, x2 - x1))

            if angle < 0:  # catch negative (clock-wise) angles
                angle = 360 + angle
    except Exception as e:
        print('p1: {}, p2: {}'.format(p1, p2))
        raise e

    return angle

def calc_mean_ip(ips_list):
    mant, minf = np.NaN, np.NaN
    if isinstance(ips_list, str): ips_list = literal_eval(ips_list)
    ants, infs = ips_list
    ants, infs = [elem for elem in ants if elem is not None], [elem for elem in infs if elem is not None]
    if len(ants)>0 and len(infs)>0:
        mant, minf = np.array(ants).mean(axis=0), np.array(infs).mean(axis=0)
    return mant, minf

def get_diff(angles1, angles2):
    diff = None
    if angles1 is not None and angles2 is not None:
        diff = abs(angles1-angles2)
    return diff

def get_differences(angles1, angles2):
    diff = np.array([abs(a-b) if a is not None and b is not None else None for a, b in zip(angles1, angles2)])
    return diff

def get_dist(p1, p2):
    # returns the euclidean distance if both points are defined, otherwise return None
    dist = None
    if (p1 is not None) and (p2 is not None):
        p1 = np.array(p1)
        p2 = np.array(p2)
        dist = np.linalg.norm(p1 - p2)
    return dist


def calc_distances(vol1, vol2, vol1ismsk=False, vol2ismsk=False, usemeanips=False):

    assert (vol1.shape == vol2.shape), 'wrong shape? vol1: {} vol2: {}'.format(vol1.shape, vol2.shape)
    if vol1ismsk:
        vol1_ants, vol1_infs = get_ip_from_mask_3d(vol1, keepdim=True, rev=True)
    else:
        vol1_ants, vol1_infs = get_ip_from_rvip_mask_3d(vol1, keepdim=True)
    if vol2ismsk:
        vol2_ants, vol2_infs = get_ip_from_mask_3d(vol2, keepdim=True, rev=True)
    else:
        vol2_ants, vol2_infs = get_ip_from_rvip_mask_3d(vol2, keepdim=True)

    # first get the mean ips, than calculate the distance
    # by this we are robust to ouliers
    if usemeanips:
        vol1_ants, vol1_infs = [elem for elem in vol1_ants if elem is not None], [elem for elem in vol1_infs if
                                                                                  elem is not None]
        vol1_ants, vol1_infs = np.array(vol1_ants, dtype=np.float), np.array(vol1_infs, dtype=np.float)
        vol1_ants, vol1_infs = [vol1_ants.mean(axis=0)], [vol1_infs.mean(axis=0)]
        vol2_ants, vol2_infs = [elem for elem in vol2_ants if elem is not None], [elem for elem in vol2_infs if
                                                                                  elem is not None]
        vol2_ants, vol2_infs = np.array(vol2_ants, dtype=np.float), np.array(vol2_infs, dtype=np.float)
        vol2_ants, vol2_infs = [vol2_ants.mean(axis=0)], [vol2_infs.mean(axis=0)]

    ant_dist = np.array([get_dist(a, b) for a, b in zip(vol1_ants, vol2_ants)])
    inf_dist = np.array([get_dist(a, b) for a, b in zip(vol1_infs, vol2_infs)])
    return ant_dist, inf_dist


def calc_dist_files(gt_f, pred_f, gtismsk=False, predismsk=False, physical=False, usemeanips=False):
    spacing = 1
    if physical:
        spacing = sitk.ReadImage(gt_f).GetSpacing()[0]  # get inplane spacing which is always square
    gt, pred = list(map(lambda x: sitk.GetArrayFromImage(sitk.ReadImage(x)), [gt_f, pred_f]))
    ant_dist, inf_dist = calc_distances(gt, pred, vol1ismsk=gtismsk, vol2ismsk=predismsk, usemeanips=usemeanips)
    # anterior distance mean, sd, inferior distance mean, sd
    ant_dist, inf_dist = np.array(ant_dist, dtype=np.float), np.array(inf_dist, dtype=np.float)
    if physical: ant_dist, inf_dist = ant_dist * spacing, inf_dist * spacing
    adm, ads, idm, ids = np.nanmean(ant_dist), np.nanstd(ant_dist), np.nanmean(inf_dist), np.nanstd(inf_dist)
    clean_result = list(map(float, [adm, ads, idm, ids]))
    return clean_result


def calc_angles2x(vol, ismsk=False, usemeanips=False):
    if ismsk:  # Note: cv2.findcontour usually returns y,x, here we want x,y
        ants, infs = get_ip_from_mask_3d(vol, keepdim=True, rev=True)
    else:
        ants, infs = get_ip_from_rvip_mask_3d(vol, keepdim=True)
    # first get the mean ips, than calculate the angle
    # by this we are robust to ouliers
    if usemeanips:
        ants, infs = [elem for elem in ants if elem is not None], [elem for elem in infs if elem is not None]
        ants, infs = np.array(ants, dtype=np.float), np.array(infs, dtype=np.float)
        ants, infs = [ants.mean(axis=0)], [infs.mean(axis=0)]
        # Note: the y-axis is inverted, in the plot the anterior is the upper, but numerically it is the lower
    angles = np.array([get_angle2x(a, b) if (a is not None and b is not None) else None for a, b in zip(ants, infs)])
    return angles


def calc_mean_angle(file_, ismsk=False, usemeanips=False):
    rvip_msk = sitk.GetArrayFromImage(sitk.ReadImage(file_))
    rvip_msk_angles = calc_angles2x(rvip_msk, ismsk=ismsk, usemeanips=usemeanips)
    rvip_msk_angles = np.array(rvip_msk_angles, dtype=np.float)
    angle_mean, angle_sd = np.nanmean(rvip_msk_angles), np.nanstd(rvip_msk_angles)
    clean_result = list(map(float, [angle_mean, angle_sd]))
    return clean_result


def calc_mean_angle_diff(gt_f, pred_f, isgtmsk=False, ispredmsk=False, usemeanips=False):
    gt, pred = list(map(lambda x: sitk.GetArrayFromImage(sitk.ReadImage(x)), [gt_f, pred_f]))
    assert (gt.shape == pred.shape)
    gt_angle, pred_angle = calc_mean_angle(gt_f, ismsk=isgtmsk, usemeanips=usemeanips), calc_mean_angle(pred_f,
                                                                                                        ismsk=ispredmsk,
                                                                                                        usemeanips=usemeanips)
    gt_angle, gt_angle_sd, pred_angle, pred_angle_sd = gt_angle[0], gt_angle[1], pred_angle[0], pred_angle[1]
    # print('mean gt_angle {}+/-{}, \nmean pred_angle {}+/-{}'.format(gt_angle, gt_angle_sd, pred_angle, pred_angle_sd))
    res = abs(gt_angle - pred_angle)
    return res, gt_angle, gt_angle_sd, pred_angle, pred_angle_sd


sorting_lambda = lambda x: int(os.path.basename(x).split('_')[0].split('patient')[1])
sorting_lambda_frame = lambda x: (
int(os.path.basename(x).split('_')[0].split('patient')[1]), int(os.path.basename(x).split('_')[1].split('frame')[1]))
sorting_lambda_frame_orig = lambda x: (int(os.path.basename(x).split('_')[0].split('patient')[1]),
                                       int(os.path.basename(x).split('_')[1].split('frame')[1].split('.')[0]))


# create one dataframe per experiment with all angle comparisons
def get_angles_as_df(files1, files2, f1ismsk=False, f2ismsk=False, suffix='pred', meanips=False):
    cols = ['angle_diff_{}'.format(suffix), 'gt_angle', 'gt_angle_sd', '{}_angle'.format(suffix),
            '{}_angle_sd'.format(suffix)]
    df_angles_temp = pd.DataFrame(columns=cols)
    for i in range(len(files1)):
        gt_f = files1[i]
        pred_f = files2[i]
        to_append = calc_mean_angle_diff(gt_f, pred_f, isgtmsk=f1ismsk, ispredmsk=f2ismsk, usemeanips=meanips)
        a_series = pd.Series(to_append, index=df_angles_temp.columns)
        df_angles_temp = df_angles_temp.append(a_series, ignore_index=True)
    return df_angles_temp


def get_dist_as_df(files1, files2, f1ismsk=False, f2ismsk=False, suffix='pred', meanips=False):
    cols = ['ant_dist_{}'.format(suffix), 'ant_dist_sd_{}'.format(suffix), 'inf_dist_{}'.format(suffix),
            'inf_dis_sd_{}'.format(suffix)]
    df_angles_temp = pd.DataFrame(columns=cols)
    for i in range(len(files1)):
        gt_f = files1[i]
        pred_f = files2[i]
        to_append = calc_dist_files(gt_f, pred_f, gtismsk=f1ismsk, predismsk=f2ismsk, physical=False,
                                    usemeanips=meanips)
        a_series = pd.Series(to_append, index=df_angles_temp.columns)
        df_angles_temp = df_angles_temp.append(a_series, ignore_index=True)
    return df_angles_temp



####### copied from EvaluationHelper
import numpy as np
import SimpleITK as sitk
import pandas as pd

from src.data.Preprocess import get_ip_from_mask_3d

# calculate the TPR with a threshold
# use this one in the module
from ast import literal_eval
def calc_tpr_thresh(gt, pred, thresh=1000, spacing=1):
    # if no threshold defined, count each rvip detected on a slice as TP

    if isinstance(gt, str) :gt = literal_eval(gt)
    if isinstance(pred, str): pred = literal_eval(pred)

    gt_ant, gt_inf = gt
    pred_ant, pred_inf = pred

    ant = 0
    inf = 0
    tp_ant = 0
    tp_inf = 0
    fn_ant = 0
    fn_inf = 0
    tpr_ant = 0
    tpr_inf = 0

    for i in range(len(gt_ant)):
        if gt_ant[i] is not None:
            if pred_ant[i] is not None:
                if get_dist(gt_ant[i], pred_ant[i] ) *spacing <= thresh: # otherwise it is a FP
                    tp_ant = tp_ant + 1
            else:
                fn_ant = fn_ant + 1
        if gt_inf[i] is not None:
            if pred_inf[i] is not None:
                if get_dist(gt_inf[i], pred_inf[i] ) *spacing <= thresh: # otherwise it is a FP
                    tp_inf = tp_inf + 1
                else:
                    fn_inf = fn_inf + 1

    if tp_ant > 0:
        tpr_ant = tp_ant /(tp_ant + fn_ant)

    if tp_inf >0:
        tpr_inf = tp_inf / (tp_inf + fn_inf)

    return tpr_ant, tpr_inf


def calc_ppv_thresh(gt, pred, thresh=1000, spacing=1):
    if isinstance(gt, str): gt = literal_eval(gt)
    if isinstance(pred, str): pred = literal_eval(pred)

    gt_ant, gt_inf = gt
    pred_ant, pred_inf = pred

    ant = 0
    inf = 0
    tp_ant = 0
    tp_inf = 0
    ppv_ant = 0
    ppv_inf = 0
    fp_inf = 0
    fp_ant = 0

    for i in range(len(gt_ant)):
        if gt_ant[i] is not None:
            if pred_ant[i] is not None:
                if get_dist(gt_ant[i], pred_ant[i]) * spacing <= thresh:  # otherwise it is a FP
                    tp_ant = tp_ant + 1
                else:
                    fp_ant = fp_ant + 1
        if gt_inf[i] is not None:
            if pred_inf[i] is not None:
                if get_dist(gt_inf[i], pred_inf[i]) * spacing <= thresh:  # otherwise it is a FP
                    tp_inf = tp_inf + 1
                else:
                    fp_inf = fp_inf + 1

        if pred_ant[i] is not None:
            if gt_ant[i] is None:  # false ant
                fp_ant = fp_ant + 1
        if pred_inf[i] is not None:
            if gt_inf[i] is None:  # false inf
                fp_inf = fp_inf + 1

    if tp_ant > 0:
        ppv_ant = (tp_ant) / (tp_ant + fp_ant)

    if tp_inf > 0:
        ppv_inf = (tp_inf) / (tp_inf + fp_inf)

    return ppv_ant, ppv_inf

# calc the angle differences slice-wise
def clean(string):
    lst = string.split()
    clean_lst = [i.replace('[','').replace(']','') for i in lst]
    clean_lst = list(filter(lambda x: x != "", clean_lst))
    clean_lst = [float(i) if i != 'None' else None for i in clean_lst]
    return clean_lst

def get_diffs(angles1,angles2, upper_bound=False):
    temp = angles1
    angles1 = clean(angles1)
    angles2 = clean(angles2)
    if upper_bound:
        diffs=[]
        for a, b in zip(angles1, angles2):
            if (a is not None) and (b is not None):
                diff = abs(a-b)
            elif a is not None and b is None:
                diff=180
            else:
                diff = np.NaN
            diffs.append(diff)
        diffs=np.array(diffs)
        #print(diffs)
    else:
        diffs = np.array([abs(a-b) if a is not None and b is not None else np.NaN for a, b in zip(angles1, angles2)])
    return np.nanmean(diffs)



def get_ip_from_rvip_file(f_name, keepdim=False, both_only=True):
    nda = sitk.GetArrayFromImage(sitk.ReadImage(f_name))
    return get_ip_from_rvip_mask_3d(nda, keepdim=keepdim, both_only=both_only)

def get_ip_from_rvip_mask_3d(msk_3d, debug=False, keepdim=False, both_only=True):
    '''
    Returns two lists of RV insertion points (y,x)-coordinates
    For a standard SAX orientation:
    the first list belongs to the anterior IP and the second to the inferior IP
    Parameters
    ----------
    msk_3d : (np.ndarray) with z,y,x
    debug : (bool) print additional info
    keepdim: (bool) returns two lists of the same length as z, slices where no RV IPs were found are represented by an tuple of None

    Returns tuple of lists with points (y,x)-coordinates
    -------

    '''
    first_ips = []
    second_ips = []
    for msk2d in msk_3d:
        try:
            first, second = get_mean_rvip_2d(msk2d, both_only=both_only)
            if (first is not None and second is not None) or keepdim:  # note the object comparison 'is' instead of '='
                first_ips.append(first)
                second_ips.append(second)
        except Exception as e:
            print(str(e))
            pass

    return first_ips, second_ips

def get_mean_rvip_2d(nda_2d, both_only=False):
    """
    Expects a numpy ndarray/ mask with at least two one-hot encoded labels,
    pixels==1 represents the anterior IP, pixels==2 the inferior IP
    returns a list with len== number of unique values in nda,
    each element is a point (y/x-coordinates) (from an image perspective) pointing the mean coordinates of this value
    usually this will return [anterior IP, inferior IP],
    could also be used for more points
    """

    assert len(nda_2d.shape) == 2, 'invalid shape: {}'.format(nda_2d.shape)
    points_dict = {"1": None, "2": None}  # pixel_value == 1 corresponds to anterior, ==2 corresponds to inferior

    labels = np.unique(nda_2d)[1:]  # ignore background

    if labels is not None:
        if both_only and len(labels) != 2: return points_dict["1"], points_dict["2"]  # return None, None
        # If both_only flag is set, then this loop computes only for the cases where both points (Ant + Inf) is available
        # Otherwise it computes for cases where labels is available

        for pixel_value in labels:
            single_ip_pixels = np.where(nda_2d == pixel_value)
            points_dict[str(int(pixel_value))] = list(
                np.array(single_ip_pixels).mean(axis=1))  # assigns to either ant or inf depending on label
    return points_dict["1"], points_dict["2"]  # return anterior, inferior


# def get_ip_from_rvip_mask_3d(msk_3d, debug=False, keepdim=False):
#     '''
#     Returns two lists of RV insertion points (y,x)-coordinates
#     For a standard SAX orientation:
#     the first list belongs to the anterior IP and the second to the inferior IP
#     Parameters
#     ----------
#     msk_3d : (np.ndarray) with z,y,x
#     debug : (bool) print additional info
#     keepdim: (bool) returns two lists of the same length as z, slices where no RV IPs were found are represented by an tuple of None
#
#     Returns tuple of lists with points (y,x)-coordinates
#     -------
#
#     '''
#     first_ips = []
#     second_ips = []
#     for msk2d in msk_3d:
#         try:
#             first, second = get_mean_rvip_2d(msk2d)
#             if (first is not None and second is not None) or keepdim:  # note the object comparison 'is' instead of '='
#                 first_ips.append(first)
#                 second_ips.append(second)
#         except Exception as e:
#             print(str(e))
#             pass
#
#     return first_ips, second_ips



# def get_mean_rvip_2d(nda_2d):
#     """
#     Expects a numpy ndarray/ mask with at least two one-hot encoded labels,
#     pixels==1 represents the anterior IP, pixels==2 the inferior IP
#     returns a list with len== number of unique values in nda,
#     each element is a point (y/x-coordinates) (from an image perspective) pointing the mean coordinates of this value
#     usually this will return [anterior IP, inferior IP],
#     could also be used for more points
#     """
#
#     assert len(nda_2d.shape) == 2, 'invalid shape: {}'.format(nda_2d.shape)
#     mean_points = []
#     labels = np.unique(nda_2d)[1:]  # ignore background
#     anterior, inferior = None, None  # return None for this slice, if no ips are available
#     if len(labels) == 2:
#         anterior, inferior = list(map(lambda x: np.where(nda_2d == x), labels))
#         anterior = np.array(anterior).mean(axis=1)
#         inferior = np.array(inferior).mean(axis=1)
#     return anterior, inferior



def isvalid(number):
    if number is None or all(np.isnan(number)):
        return False
    else:
        return True


from math import atan2, degrees


def get_angle2x(p1, p2):
    '''
    Calc the angle between two points with a defined order
    p1==anterior, p2==inferior
    (anterior and inferior IPS) and the x-axis
    we expect point as y,x tuple
    Here we calculate always the angle between the x-axis anti-clock-wise and the ip-connection line
    Parameters
    ----------
    p1 : tuple y,x
    p2 : tuple y,x

    Returns angle in degree, None if at least one of the points are None
    -------

    '''
    try:
        angle = None
        if (np.isfinite(p1).all() and np.isfinite(p2).all()):
            y1, x1, y2, x2 = p1[0], p1[1], p2[0], p2[1]
            angle = degrees(atan2(y2 - y1, x2 - x1))

            if angle < 0:  # catch negative (clock-wise) angles
                angle = 360 + angle
    except Exception as e:
        print('p1: {}, p2: {}'.format(p1, p2))
        raise e

    return angle

def get_dist(p1, p2):
    # returns the euclidean distance if both points are defined, otherwise return None
    dist = None
    if (p1 is not None) and (p2 is not None):
        p1 = np.array(p1)
        p2 = np.array(p2)
        dist = np.linalg.norm(p1 - p2)
    return dist


# this version goes into the module, check the threshold, to make sure!!!
def get_distances(ips1, ips2, spacing=1, threshold=None):
    vol1_ants, vol1_infs = ips1
    vol2_ants, vol2_infs = ips2
    ant_dist = [get_dist(a, b) * spacing if a is not None and b is not None else None for a, b in
                zip(vol1_ants, vol2_ants)]
    inf_dist = [get_dist(a, b) * spacing if a is not None and b is not None else None for a, b in
                zip(vol1_infs, vol2_infs)]

    if threshold is not None:
        ant_dist = [dist if dist is not None and dist <= threshold else None for dist in ant_dist]
        inf_dist = [dist if dist is not None and dist <= threshold else None for dist in inf_dist]

    return np.array(ant_dist), np.array(inf_dist)

def get_mean_dist(dist_for_all_slices):
    # It takes the mean of all the values in the array that are not None, if all the values are None then returns None
    dist_for_all_slices = np.array(dist_for_all_slices)
    dist_for_all_slices = dist_for_all_slices[dist_for_all_slices!= None]

    if len(dist_for_all_slices) > 0: return np.mean(dist_for_all_slices)
    else: return None


def get_distances_upper_bound(ips1, ips2, spacing=1, dim=224):
    # ips1 is expected GT, ips2 is expected pred
    vol1_ants, vol1_infs = ips1
    vol2_ants, vol2_infs = ips2

    ant_dist = [None] * len(vol1_ants)
    inf_dist = [None] * len(vol1_infs)

    def upper_bound(point):
        return max([get_dist(point, coord) * spacing for coord in [(0, 0), (0, dim), (dim, 0), (dim, dim)]])

    for i, (a, b) in enumerate(zip(vol1_ants, vol2_ants)):
        if a is not None and b is not None:
            ant_dist[i] = get_dist(a, b) * spacing
        elif a is not None and b is None:
            ant_dist[i] = upper_bound(a)

    for i, (a, b) in enumerate(zip(vol1_infs, vol2_infs)):
        if a is not None and b is not None:
            inf_dist[i] = get_dist(a, b) * spacing
        elif a is not None and b is None:
            inf_dist[i] = upper_bound(a)

    return np.array(ant_dist), np.array(inf_dist)



def evaluate_cv_save(exp_path, data_path):
    # ------------------------------------------ import helpers
    # this should import glob, os, and some other standard libs to keep this cell clean
    # local imports
    from src.utils.Utils_io import Console_and_file_logger, init_config
    from src.visualization.Visualize import show_2D_or_3D
    from src.data.Preprocess import get_ip_from_mask_3d

    # load all necessary file names
    data_root = data_path
    path_to_exp = exp_path
    # this will collect the predictions within the 4 cv subfolders
    exp_path = os.path.join(path_to_exp, '*/*/')
    io_files = sorted(glob.glob(os.path.join(data_root, 'io/', '*rvip.nrrd')), key=sorting_lambda_frame)
    pred_files = sorted(glob.glob(os.path.join(exp_path, 'pred', '*msk.nrrd')), key=sorting_lambda)
    gt_files = sorted(glob.glob(os.path.join(exp_path, 'gt', '*msk.nrrd')), key=sorting_lambda)
    cmr_files = sorted(glob.glob(os.path.join(exp_path, 'pred', '*cmr.nrrd')), key=sorting_lambda)
    print('io files: ', len(io_files))
    print('pred fies: ', len(pred_files))
    print('gt files: ', len(gt_files))
    print('cmr files: ', len(cmr_files))

    #  original masks
    orig_msk_files = sorted(glob.glob(os.path.join(data_root, 'original', '*/*frame*gt.nii.gz')),
                            key=sorting_lambda_frame)
    print('original msk files: ', len(orig_msk_files))
    # original cmr
    orig_cmr_files = sorted(glob.glob(os.path.join(data_root, 'original', '*/*frame[0-9][0-9].nii.gz')),
                            key=sorting_lambda_frame_orig)
    print('original cmr files: ', len(orig_cmr_files))

    # load acdc metadata as df
    from src.data.Dataset import get_acdc_dataset_as_df
    df = get_acdc_dataset_as_df(os.path.join(data_root, 'original'))
    df = df.loc[df['phase'].isin(['ed', 'es'])]
    df.reset_index(inplace=True, drop=True)

    # prepare some instructions
    files_ = [pred_files, io_files, orig_msk_files]
    ismsks = [False, False, True]
    suffixes = ['pred', 'io', 'orig_msk']
    use_the_mean_rvip = True
    # create a df 200 x 15 with all angles
    dfs = [get_angles_as_df(gt_files, f_, f2ismsk=b, suffix=s, meanips=use_the_mean_rvip) for f_, b, s in
           zip(files_, ismsks, suffixes)]
    df_angles = pd.concat(dfs, axis=1)
    # create a df 200 x 12 with all distances
    dfs = [get_dist_as_df(gt_files, f_, f2ismsk=b, suffix=s, meanips=use_the_mean_rvip) for f_, b, s in
           zip(files_, ismsks, suffixes)]
    df_dists = pd.concat(dfs, axis=1)
    # combine angles and distances --> 200,27
    df_eval = pd.concat([df_angles, df_dists], axis=1)

    # extend the dataframe by patient id, phase and pathology --> 200, 33
    df_eval['pred_files'], df_eval['io_files'], df_eval['orig_msk_files'] = pred_files, io_files, orig_msk_files
    df_eval['patient'] = df_eval['pred_files'].map(lambda x: os.path.basename(x).split('_')[0])
    df_eval['phase'] = df_eval['pred_files'].map(lambda x: os.path.basename(x).split('_')[1])
    df_eval['pathology'] = df['pathology']
    df_eval = df_eval.loc[:, ~df_eval.columns.duplicated()]
    df_eval.to_csv(os.path.join(path_to_exp, 'df_eval.csv'), index=False)
    print('evaluation done for {}'.format({exp_path}))
    return

def evaluate_cv(exp_path, data_path):
    # load all necessary file names
    data_root = data_path
    path_to_exp = exp_path
    # this will collect the predictions within the 4 cv subfolders
    exp_path = os.path.join(path_to_exp, '*/*/')
    io_files = sorted(glob.glob(os.path.join(data_root, 'io/', '*rvip.nrrd')), key=sorting_lambda_frame)
    pred_files = sorted(glob.glob(os.path.join(exp_path, 'pred', '*msk.nrrd')), key=sorting_lambda)
    gt_files = sorted(glob.glob(os.path.join(exp_path, 'gt', '*msk.nrrd')), key=sorting_lambda)
    cmr_files = sorted(glob.glob(os.path.join(exp_path, 'pred', '*cmr.nrrd')), key=sorting_lambda)
    print('io files: ', len(io_files))
    print('pred fies: ', len(pred_files))
    print('gt files: ', len(gt_files))
    print('cmr files: ', len(cmr_files))

    #  original masks
    orig_msk_files = sorted(glob.glob(os.path.join(data_root, 'original', '*/*frame*gt.nii.gz')),
                            key=sorting_lambda_frame)
    print('original msk files: ', len(orig_msk_files))
    # original cmr
    orig_cmr_files = sorted(glob.glob(os.path.join(data_root, 'original', '*/*frame[0-9][0-9].nii.gz')),
                            key=sorting_lambda_frame_orig)
    print('original cmr files: ', len(orig_cmr_files))

    # load acdc metadata as df
    from src.data.Dataset import get_acdc_dataset_as_df
    df_raw = get_acdc_dataset_as_df(os.path.join(data_root, 'original'))
    df_raw = df_raw.loc[df_raw['phase'].isin(['ed', 'es'])]
    df_raw.reset_index(inplace=True, drop=True)

    # create a df with the filenames and the metadata
    # extend the dataframe by patient id, phase and pathology --> 200, 33
    df_eval = pd.DataFrame()
    df_eval['files_pred'], df_eval['files_io'], df_eval['files_orig_msk'],df_eval['files_gt']  = pred_files, io_files, orig_msk_files, gt_files
    df_eval['patient'] = df_eval['files_pred'].map(lambda x: os.path.basename(x).split('_')[0])
    df_eval['phase'] = df_eval['files_pred'].map(lambda x: os.path.basename(x).split('_')[1])
    df_eval['pathology'] = df_raw['pathology']
    df_eval = df_eval.loc[:, ~df_eval.columns.duplicated()]

    # Extract cmr related metadata such as the spacing
    df_eval['spacing'] = df_eval['files_gt'].map(lambda x : sitk.ReadImage(x).GetSpacing())
    df_eval['inplane_spacing'] = df_eval['spacing'].map(lambda x: x[0])

    # get the rvips for each file
    df_eval['ips_pred'] = df_eval['files_pred'].map(lambda x : get_ip_from_rvip_file(x,keepdim=True))
    df_eval['ips_gt'] = df_eval['files_gt'].map(lambda x: get_ip_from_rvip_file(x, keepdim=True))
    df_eval['ips_io'] = df_eval['files_io'].map(lambda x: get_ip_from_rvip_file(x, keepdim=True))
    df_eval['ips_orig_msk'] = df_eval['files_orig_msk'].map(lambda x: get_ip_from_ventriclemsk_file(x, keepdim=True))
    # mean ips and mean angle
    df_eval['mips_pred'] = df_eval['ips_pred'].map(lambda x: calc_mean_ip(x))
    df_eval['mips_gt'] = df_eval['ips_gt'].map(lambda x: calc_mean_ip(x))
    df_eval['mips_io'] = df_eval['ips_io'].map(lambda x: calc_mean_ip(x))
    df_eval['mips_orig_msk'] = df_eval['ips_orig_msk'].map(lambda x: calc_mean_ip(x))

    df_eval['mangle_pred'] = df_eval['mips_pred'].map(lambda x: get_angle2x(x[0],x[1]))
    df_eval['mangle_gt'] = df_eval['mips_gt'].map(lambda x: get_angle2x(x[0], x[1]))
    df_eval['mangle_io'] = df_eval['mips_io'].map(lambda x: get_angle2x(x[0], x[1]))
    df_eval['mangle_orig_msk'] = df_eval['mips_orig_msk'].map(lambda x: get_angle2x(x[0], x[1]))

    df_eval['mdiffs_gtpred'] = df_eval[['mangle_gt', 'mangle_pred']].apply(lambda x: get_diff(x['mangle_gt'], x['mangle_pred']), axis=1)
    df_eval['mdiffs_gtio'] = df_eval[['mangle_gt', 'mangle_io']].apply(
        lambda x: get_diff(x['mangle_gt'], x['mangle_io']), axis=1)
    df_eval['mdiffs_gtorig'] = df_eval[['mangle_gt', 'mangle_orig_msk']].apply(
        lambda x: get_diff(x['mangle_gt'], x['mangle_orig_msk']), axis=1)

    df_eval['mdists_ant_gtpred']= df_eval[['mips_gt', 'mips_pred']].apply(
        lambda x: get_dist(x['mips_gt'][0], x['mips_pred'][0]), axis=1)

    df_eval['mdists_inf_gtpred'] = df_eval[['mips_gt', 'mips_pred']].apply(
        lambda x: get_dist(x['mips_gt'][1], x['mips_pred'][1]), axis=1)


    df_eval['mdists_ant_gtio'] = df_eval[['mips_gt', 'mips_io']].apply(
        lambda x: get_dist(x['mips_gt'][0], x['mips_io'][0]), axis=1)
    df_eval['mdists_inf_gtio'] = df_eval[['mips_gt', 'mips_io']].apply(
        lambda x: get_dist(x['mips_gt'][1], x['mips_io'][1]), axis=1)

    df_eval['mdists_ant_gtorig'] = df_eval[['mips_gt', 'mips_orig_msk']].apply(
        lambda x: get_dist(x['mips_gt'][0], x['mips_orig_msk'][0]), axis=1)
    df_eval['mdists_inf_gtorig'] = df_eval[['mips_gt', 'mips_orig_msk']].apply(
        lambda x: get_dist(x['mips_gt'][1], x['mips_orig_msk'][1]), axis=1)

    # mean distances in mm
    df_eval['mdists_ant_gtpred'] = df_eval['mdists_ant_gtpred'] * df_eval['inplane_spacing']
    df_eval['mdists_inf_gtpred'] = df_eval['mdists_inf_gtpred'] * df_eval['inplane_spacing']
    df_eval['mdists_ant_gtio'] = df_eval['mdists_ant_gtio'] * df_eval['inplane_spacing']
    df_eval['mdists_inf_gtio'] = df_eval['mdists_inf_gtio'] * df_eval['inplane_spacing']
    df_eval['mdists_ant_gtorig'] = df_eval['mdists_ant_gtorig'] * df_eval['inplane_spacing']
    df_eval['mdists_inf_gtorig'] = df_eval['mdists_inf_gtorig'] * df_eval['inplane_spacing']


    # calculate the angles for each file
    df_eval['angles_pred'] = df_eval['ips_pred'].map(lambda x : get_angles2x(x))
    df_eval['angles_gt'] = df_eval['ips_gt'].map(lambda x: get_angles2x(x))
    df_eval['angles_io'] = df_eval['ips_io'].map(lambda x: get_angles2x(x))
    df_eval['angles_orig_msk'] = df_eval['ips_orig_msk'].map(lambda x: get_angles2x(x))

    # calculate the distances for the ips
    df_eval['dists_ant_gtpred'], df_eval['dists_inf_gtpred'] = zip(
        *df_eval[['ips_gt', 'ips_pred', 'inplane_spacing']].apply(
            lambda x : get_distances(x['ips_gt'], x['ips_pred'], x['inplane_spacing']), axis=1))
    df_eval['dists_ant_gtio'], df_eval['dists_inf_gtio'] = zip(
        *df_eval[['ips_gt', 'ips_io', 'inplane_spacing']].apply(
            lambda x: get_distances(x['ips_gt'], x['ips_io'], x['inplane_spacing']), axis=1))
    df_eval['dists_ant_gtorig'], df_eval['dists_inf_gtorig'] = zip(
        *df_eval[['ips_gt', 'ips_orig_msk', 'inplane_spacing']].apply(
            lambda x: get_distances(x['ips_gt'], x['ips_orig_msk'], x['inplane_spacing']), axis=1))

    # calculate the differences for the angles
    df_eval['diffs_gtpred'] = df_eval[['angles_gt', 'angles_pred']].apply(lambda x: get_differences(x['angles_gt'], x['angles_pred']), axis=1)
    df_eval['diffs_gtio'] = df_eval[['angles_gt', 'angles_io']].apply(
        lambda x: get_differences(x['angles_gt'], x['angles_io']), axis=1)
    df_eval['diffs_gtorig'] = df_eval[['angles_gt', 'angles_orig_msk']].apply(
        lambda x: get_differences(x['angles_gt'], x['angles_orig_msk']), axis=1)
    df_eval['EXP'] = [path_to_exp]*len(df_eval)

    # extension with TPR
    # TPR
    df_eval['tpr_ant'], df_eval['tpr_inf'] = list(zip(*df_eval[['ips_gt', 'ips_pred']].apply(lambda x: calc_tpr_thresh(x[0], x[1]), axis=1)))
    df_eval['tpr_ant_io'], df_eval['tpr_inf_io'] = list(
        zip(*df_eval[['ips_gt', 'ips_io']].apply(lambda x: calc_tpr_thresh(x[0], x[1]), axis=1)))
    df_eval['tpr_ant_msk'], df_eval['tpr_inf_msk'] = list(
        zip(*df_eval[['ips_gt', 'ips_orig_msk']].apply(lambda x: calc_tpr_thresh(x[0], x[1]), axis=1)))

    # PPV
    df_eval['ppv_ant'], df_eval['ppv_inf'] = list(zip(*df_eval[['ips_gt', 'ips_pred']].apply(lambda x: calc_ppv_thresh(x[0], x[1]), axis=1)))
    df_eval['ppv_ant_io'], df_eval['ppv_inf_io'] = list(
        zip(*df_eval[['ips_gt', 'ips_io']].apply(lambda x: calc_ppv_thresh(x[0], x[1]), axis=1)))
    df_eval['ppv_ant_msk'], df_eval['ppv_inf_msk'] = list(
        zip(*df_eval[['ips_gt', 'ips_orig_msk']].apply(lambda x: calc_ppv_thresh(x[0], x[1]), axis=1)))


    # extension according to localisation and detection based metrics
    df_eval['ips_pred_single_also'] = df_eval['files_pred'].map(
        (lambda x: get_ip_from_rvip_file(x, keepdim=True, both_only=False)))

    # *************** Slice based
    # TPR
    df_eval['tpr_ant'], df_eval['tpr_inf'] = list(
        zip(*df_eval[['ips_gt', 'ips_pred']].apply(lambda x: calc_tpr_thresh(x[0], x[1]), axis=1)))
    df_eval['tpr_ant_io'], df_eval['tpr_inf_io'] = list(
        zip(*df_eval[['ips_gt', 'ips_io']].apply(lambda x: calc_tpr_thresh(x[0], x[1]), axis=1)))
    df_eval['tpr_ant_msk'], df_eval['tpr_inf_msk'] = list(
        zip(*df_eval[['ips_gt', 'ips_orig_msk']].apply(lambda x: calc_tpr_thresh(x[0], x[1]), axis=1)))

    # PPV
    df_eval['ppv_ant'], df_eval['ppv_inf'] = list(
        zip(*df_eval[['ips_gt', 'ips_pred']].apply(lambda x: calc_ppv_thresh(x[0], x[1]), axis=1)))
    df_eval['ppv_ant_io'], df_eval['ppv_inf_io'] = list(
        zip(*df_eval[['ips_gt', 'ips_io']].apply(lambda x: calc_ppv_thresh(x[0], x[1]), axis=1)))
    df_eval['ppv_ant_msk'], df_eval['ppv_inf_msk'] = list(
        zip(*df_eval[['ips_gt', 'ips_orig_msk']].apply(lambda x: calc_ppv_thresh(x[0], x[1]), axis=1)))

    # *************** Point based
    df_eval['tpr_ant_point'], df_eval['tpr_inf_point'] = list(
        zip(*df_eval[['ips_gt', 'ips_pred_single_also']].apply(lambda x: calc_tpr_thresh(x[0], x[1]), axis=1)))
    df_eval['ppv_ant_point'], df_eval['ppv_inf_point'] = list(
        zip(*df_eval[['ips_gt', 'ips_pred_single_also']].apply(lambda x: calc_ppv_thresh(x[0], x[1]), axis=1)))

    # *************** Point based with threshold=15mm
    df_eval['tpr_ant_point_th15'], df_eval['tpr_inf_point_th15'] = list(
        zip(*df_eval[['ips_gt', 'ips_pred_single_also', 'inplane_spacing']].apply(
            lambda x: calc_tpr_thresh(x['ips_gt'], x['ips_pred_single_also'], thresh=15, spacing=x['inplane_spacing']),
            axis=1)))

    df_eval['ppv_ant_point_th15'], df_eval['ppv_inf_point_th15'] = list(
        zip(*df_eval[['ips_gt', 'ips_pred_single_also', 'inplane_spacing']].apply(
            lambda x: calc_ppv_thresh(x['ips_gt'], x['ips_pred_single_also'], thresh=15, spacing=x['inplane_spacing']),
            axis=1)))

    # Compute the distances and mdist for gt vs pred_single_also
    df_eval['mips_pred_single_also'] = df_eval['ips_pred_single_also'].map(lambda x: calc_mean_ip(x))
    df_eval['mdists_ant_gtpred_single_also'] = df_eval[['mips_gt', 'mips_pred_single_also']].apply(
        lambda x: get_dist(x['mips_gt'][0], x['mips_pred_single_also'][0]), axis=1)

    df_eval['mdists_inf_gtpred_single_also'] = df_eval[['mips_gt', 'mips_pred_single_also']].apply(
        lambda x: get_dist(x['mips_gt'][1], x['mips_pred_single_also'][1]), axis=1)

    # mean distances in mm
    df_eval['mdists_ant_gtpred_single_also'] = df_eval['mdists_ant_gtpred_single_also'] * df_eval['inplane_spacing']
    df_eval['mdists_inf_gtpred_single_also'] = df_eval['mdists_inf_gtpred_single_also'] * df_eval['inplane_spacing']

    # For 2 IPS detected only
    df_eval['dists_ant_gtpred'], df_eval['dists_inf_gtpred'] = zip(
        *df_eval[['ips_gt', 'ips_pred', 'inplane_spacing']].apply(
            lambda x: get_distances(x['ips_gt'], x['ips_pred'], x['inplane_spacing']), axis=1))

    df_eval['mdists_ant_gtpred_slice_wise'] = df_eval['dists_ant_gtpred'].map(lambda x: get_mean_dist(x))
    df_eval['mdists_inf_gtpred_slice_wise'] = df_eval['dists_inf_gtpred'].map(lambda x: get_mean_dist(x))

    # For 1 IPS detected also
    df_eval['dists_ant_gtpred_single_also'], df_eval['dists_inf_gtpred_single_also'] = zip(
        *df_eval[['ips_gt', 'ips_pred_single_also', 'inplane_spacing']].apply(
            lambda x: get_distances(x['ips_gt'], x['ips_pred_single_also'], x['inplane_spacing']), axis=1))

    df_eval['mdists_ant_gtpred_slice_wise_single_also'] = df_eval['dists_ant_gtpred_single_also'].map(lambda x: get_mean_dist(x))
    df_eval['mdists_inf_gtpred_slice_wise_single_also'] = df_eval['dists_inf_gtpred_single_also'].map(lambda x: get_mean_dist(x))

    # For 2 IPS detected only
    df_eval['dists_ant_gtpred_up'], df_eval['dists_inf_gtpred_up'] = zip(
        *df_eval[['ips_gt', 'ips_pred', 'inplane_spacing']].apply(
            lambda x: get_distances_upper_bound(x['ips_gt'], x['ips_pred'], x['inplane_spacing']), axis=1))

    df_eval['mdists_ant_gtpred_slice_wise_up'] = df_eval['dists_ant_gtpred_up'].map(lambda x: get_mean_dist(x))
    df_eval['mdists_inf_gtpred_slice_wise_up'] = df_eval['dists_inf_gtpred_up'].map(lambda x: get_mean_dist(x))

    # For 1 IPS detected also
    df_eval['dists_ant_gtpred_single_also_up'], df_eval['dists_inf_gtpred_single_also_up'] = zip(
        *df_eval[['ips_gt', 'ips_pred_single_also', 'inplane_spacing']].apply(
            lambda x: get_distances_upper_bound(x['ips_gt'], x['ips_pred_single_also'], x['inplane_spacing']), axis=1))

    df_eval['mdists_ant_gtpred_slice_wise_single_also_up'] = df_eval['dists_ant_gtpred_single_also_up'].map(
        lambda x: get_mean_dist(x))
    df_eval['mdists_inf_gtpred_slice_wise_single_also_up'] = df_eval['dists_inf_gtpred_single_also_up'].map(
        lambda x: get_mean_dist(x))



    # save df
    df_eval.to_csv(os.path.join(path_to_exp, 'df_eval.csv'), index=False)

    print('evaluation done for {}'.format({exp_path}))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='evaluate the cv of a rvip detection model')

    # usually these two parameters should encapsulate all experiment parameters
    parser.add_argument('-exp', action='store', default=None)
    parser.add_argument('-data', action='store', default=None)

    results = parser.parse_args()
    print('given parameters: {}'.format(results))
    evaluate_cv(results.exp, results.data)
    try:
        pass
        #evaluate_cv(results.exp, results.data)
    except Exception as e:
        print(e)
    exit()