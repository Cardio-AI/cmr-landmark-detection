import pandas as pd

from src.data.Dataset import save_all_3d_vols_new
from src.data.Postprocess import undo_generator_steps


def pred_fold(config, debug=True):
    # make sure all neccessary params in config are set
    # if not set them with default values
    from src.utils.Tensorflow_helper import choose_gpu_by_id
    # ------------------------------------------define GPU id/s to use
    GPU_IDS = config.get('GPU_IDS', '0,1')
    GPUS = choose_gpu_by_id(GPU_IDS)
    print(GPUS)
    # ------------------------------------------ import helpers
    # this should import glob, os, and many other standard libs
    from tensorflow.python.client import device_lib
    import tensorflow as tf
    tf.get_logger().setLevel('FATAL')
    import gc, logging, os, datetime, re
    from logging import info

    # local imports
    from src.utils.Utils_io import Console_and_file_logger, init_config, ensure_dir
    from src.utils.KerasCallbacks import get_callbacks
    from src.data.Dataset import get_trainings_files
    from src.data.Generators import DataGenerator
    from src.models.Unets import create_unet
    import numpy as np
    from src.data.Dataset import save_gt_and_pred

    # import external libs
    from time import time
    import SimpleITK as sitk
    from scipy import ndimage

    import os

    # make all config params known to the local namespace
    locals().update(config)

    # overwrite the experiment names and paths, so that each cv gets an own sub-folder
    EXPERIMENT = config.get('EXPERIMENT')
    FOLD = config.get('FOLD')

    EXPERIMENT = '{}_f{}'.format(EXPERIMENT, FOLD)
    timestemp = str(datetime.datetime.now().strftime(
        "%Y-%m-%d_%H_%M"))  # add a timestep to each project to make repeated experiments unique

    DATA_PATH_SAX = config.get('DATA_PATH_SAX')
    DF_FOLDS = config.get('DF_FOLDS')
    EPOCHS = config.get('EPOCHS', 100)

    Console_and_file_logger(path=config.get('EXP_PATH'), log_lvl=logging.INFO)
    # get kfolded data from DATA_ROOT and subdirectories
    # Load SAX volumes
    x_train_sax, y_train_sax, x_val_sax, y_val_sax = get_trainings_files(data_path=DATA_PATH_SAX,
                                                                         path_to_folds_df=DF_FOLDS,
                                                                         fold=FOLD)
    path_to_orig = config['DATA_PATH_ORIG']
    # load all orig cmr files, if path is given
    import glob
    orig_given = False
    orig_cmr_files = sorted(glob.glob(os.path.join(path_to_orig, '*/*frame[0-9][0-9].nii.gz')))
    logging.info('Found {} orig 3D CMR images'.format(len(orig_cmr_files)))
    if len(orig_cmr_files)>0:
        orig_given=True

    logging.info('SAX train CMR: {}, SAX train masks: {}'.format(len(x_train_sax), len(y_train_sax)))
    logging.info('SAX val CMR: {}, SAX val masks: {}'.format(len(x_val_sax), len(y_val_sax)))

    t0 = time()
    try:
        # load the model, to make sure we use the same as later for the evaluations
        model = create_unet(config)
        model.load_weights(os.path.join(config['MODEL_PATH'], 'model.h5'))
        logging.info('loaded model weights as h5 file')

        pred_path = os.path.join(config.get('EXP_PATH'), 'pred')
        ensure_dir(pred_path)
        gt_path = os.path.join(config.get('EXP_PATH'), 'gt')
        ensure_dir(gt_path)

        # create a generator with idempotent behaviour (no shuffle etc.)
        # make sure we save always the same patient
        pred_config = config.copy()
        pred_config['SHUFFLE'] = False
        pred_config['AUGMENT'] = False
        pred_config['BATCHSIZE'] = 1
        pred_config['HIST_MATCHING'] = False

        df = pd.read_csv(DF_FOLDS)
        df = df[df['fold'] == FOLD]
        df = df[df['modality'] == 'test']

        # filter a list of filenames by a patient id, this is necessary as the filepath in our df differs from the real filenames
        def filter_by_patient_id(p_id, f_names):
            return [elem for elem in f_names if p_id in elem]

        # show only data on 'unique' patients to sum up folds and slices
        for p in sorted(df['patient'].unique()):  # for each patient
            info(p)  # shows which patient we are at
            # load files and masks for given patient
            files_ = filter_by_patient_id(p, x_val_sax)
            masks_ = filter_by_patient_id(p, y_val_sax)

            info(len(files_))  # shows amount of slices for each patient
            # collect all files for this patient
            # split in ED and ES, using the fact that both have the same amount of slices and the data is sorted.
            ed_f = files_[:len(files_) // 2]
            es_f = files_[len(files_) // 2:]
            ed_m = masks_[:len(masks_) // 2]
            es_m = masks_[len(masks_) // 2:]
            f_ = [ed_f, es_f]
            m_ = [ed_m, es_m]
            phases = ['ED', 'ES']
            assert (len(ed_m) == len(ed_f)), 'number of images and masks should be the same, something went wrong'
            info('length of ed_f ' + str(len(ed_f)))
            info('length of es_f ' + str(len(es_f)))
            # print('this is ed_f ' + ed_f)
            # print('this is es_f ' + es_f)

            # the following is looped twice so both phases, ED and ES are processed.
            for p_ in range(2):
                phase_cmr_files = f_[p_]
                phase_mask_files = m_[p_]
                current_phase = phases[p_]
                info('patient: {}, phase: {}, files: {}'.format(p, current_phase, len(phase_cmr_files)))

                # create validation generator just for the given patient and fold.
                # This means that each patient requires two generators.
                # This trick allows us to align the info from the .csv with the data from the generators.
                validation_generator = DataGenerator(phase_cmr_files, phase_mask_files, config=pred_config)

                # get cmr mask and save in a numpy.stack
                gts = np.stack([np.squeeze(y) for x, y in validation_generator])
                logging.info('groundtruth shape' + str(gts.shape))
                # get cmr image and save in a numpy stack
                gts_cmr = np.stack([np.squeeze(x) for x, y in validation_generator])
                logging.info('original cmr shape' + str(gts_cmr.shape))

                # predict on the validation generator
                preds = model.predict(validation_generator)
                logging.info(preds.shape)

                # upper_RVIP/anterior = 1, lower_RVIP/inferior == 2. Corresponds to annotation guide.
                # transform to int representation (one-hot-encoded)
                # create data based on ground-truth
                gts_flat = np.zeros((gts.shape[:-1]))
                gts_flat[gts[..., 0] > 0.5] = 1
                gts_flat[gts[..., 1] > 0.5] = 2

                # create data based on predictions
                preds_flat = np.zeros((gts.shape[:-1]))
                preds_flat[preds[..., 0] > 0.5] = 1
                preds_flat[preds[..., 1] > 0.5] = 2

                # keep only the biggest connected component from a 2D perspective
                if config.get('CC_FILTER', False): # usually this is better
                    from src.data.Postprocess import clean_3d_prediction_2d_cc
                    preds_flat = clean_3d_prediction_2d_cc(preds_flat)

                info(gts_flat.shape)
                info(preds_flat.shape)
                info(gts_cmr.shape)

                if orig_given:
                    temp_orig_f = filter_by_patient_id(p, orig_cmr_files)[0]
                    temp_orig = sitk.ReadImage(temp_orig_f)
                    gt_sitks = undo_generator_steps(gts_flat.astype(np.uint8), config, sitk.sitkNearestNeighbor, temp_orig)
                    pred_sitks = undo_generator_steps(preds_flat.astype(np.uint8), config, sitk.sitkNearestNeighbor, temp_orig)
                    gt_cmr_sitks = undo_generator_steps(np.stack(gts_cmr, axis=0), config, sitk.sitkNearestNeighbor, temp_orig)

                else: # no original cmr given, save the files with the spacing given by the cfg
                    # Read image data from Array using sitk library
                    gt_sitks = sitk.GetImageFromArray(gts_flat.astype(np.uint8))
                    pred_sitks = sitk.GetImageFromArray(preds_flat.astype(np.uint8))
                    gt_cmr_sitks = sitk.GetImageFromArray(np.stack(gts_cmr, axis=0))
                    exp_spacing = tuple(reversed(pred_config.get('SPACING')))
                    exp_spacing = (*exp_spacing, 10) # we should use the original spacing in Z. But for the in-plane angle/evaluation it makes no difference
                    _ = list(map(lambda x: x.SetSpacing(exp_spacing), [gt_sitks, pred_sitks, gt_cmr_sitks]))

                # Writing images to storage from previously loaded images.
                sitk.WriteImage(gt_sitks, os.path.join(gt_path, '{}_{}_msk.nrrd'.format(p, current_phase)))
                sitk.WriteImage(pred_sitks, os.path.join(pred_path, '{}_{}_msk.nrrd'.format(p, current_phase)))
                sitk.WriteImage(gt_cmr_sitks, os.path.join(pred_path, '{}_{}_cmr.nrrd'.format(p, current_phase)))

        logging.info('done! Check the folders \n{} and \n{} for files'.format(gt_path, pred_path))

        # end new version

    except Exception as e:
        logging.error(e)

    # free as much memory as possible
    del validation_generator
    del model
    gc.collect()

    logging.info('pred on fold {} finished after {:0.3f} sec'.format(FOLD, time() - t0))
    return True


def main(args=None):
    # ------------------------------------------define logging and working directory
    # import the packages inside this function enables to train on different folds
    from ProjectRoot import change_wd_to_project_root
    change_wd_to_project_root()
    import sys, os, datetime
    sys.path.append(os.getcwd())
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # ------------------------------------------define GPU id/s to use, if given

    # local imports
    from src.utils.Utils_io import Console_and_file_logger, init_config
    # import external libs
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    import cv2
    import logging
    from logging import info

    EXPERIMENTS_ROOT = 'exp/'

    if args.exp:
        import json
        cfg = os.path.join(args.exp, 'config/config.json')
        print('config given: {}'.format(cfg))
        # load the experiment config
        with open(cfg, encoding='utf-8') as data_file:
            config = json.loads(data_file.read())

            EXPERIMENT = config.get('EXPERIMENT', 'UNDEFINED')
            Console_and_file_logger(args.exp, logging.INFO)
            info('Loaded config for experiment: {}'.format(EXPERIMENT))

            # make relative paths absolute
            config['MODEL_PATH'] = os.path.join(args.exp, 'model/')
            config['EXP_PATH'] = args.exp

            # Load SAX volumes
            # cluster to local data mapping
    if args.data:
        data_root = args.data
        config['DATA_PATH_SAX'] = os.path.join(data_root, '2D')
        df_folds = os.path.join(data_root, 'df_kfold.csv')
        if os.path.isfile(df_folds) :
            config['DF_FOLDS'] = df_folds
        else :
            config['DF_FOLDS'] = None
        config['DATA_PATH_ORIG'] = os.path.join(data_root, 'original')



    pred_fold(config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='train a phase registration model')

    # usually these two parameters should encapsulate all experiment parameters
    parser.add_argument('-exp', action='store', default=None)
    parser.add_argument('-data', action='store', default=None)

    results = parser.parse_args()
    print('given parameters: {}'.format(results))

    try:
        main(results)
    except Exception as e:
        print(e)
    exit()