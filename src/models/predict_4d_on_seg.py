import json
import logging
import sys, os

import numpy as np

from src.utils.Utils_io import ensure_dir

sys.path.append(os.getcwd())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('FATAL')
from src.data.Dataset import describe_sitk
from src.data.Generators import DataGenerator, sliceable
import glob
import SimpleITK as sitk
import pandas as pd



# one function to evaluate a cv
def predict_4d_on_2d_cv(path_to_exp ,list_of_4d, export_suffix='pred_4d_cv_test'):

    # get all configs
    cfgs = list(sorted(glob.glob(os.path.join(path_to_exp, 'f*/config/config.json'))))
    logging.info('found: {} cfgs.'.format(len(cfgs)))
    logging.info('got {} files for inference'.format(len(list_of_4d)))
    assert len(cfgs ) >0, 'no cfgs found please check the path_to_exp parameter: {}'.format(path_to_exp)

    # for each config/fold call inference with
    # filtered 4d files, cfg, export_path_full
    for cfg in cfgs:
        # load the experiment config
        with open(cfg, encoding='utf-8') as data_file:
            config = json.loads(data_file.read())
        # globals().update(config)

        fold = config.get('FOLD')
        df_folds = config.get('DF_FOLDS')

        df = pd.read_csv(df_folds)
        temp = df[df['fold'] == fold]
        temp = temp[temp['modality']=='test']
        patients = temp['patient'].unique()
        logging.info('expect {} patients for validation in fold {}'.format(len(patients), fold))
        # filter the list of 4D files by a list of substrings (patient ids of val split in this fold)
        def Filter(string, substr):
            return [str for str in string if
                    any(sub.lower() in str.lower() for sub in substr)]
        files_filtered = Filter(list_of_4d ,patients)
        logging.info('4d files filtered: {}'.format(len(files_filtered)))
        if len(files_filtered)==0: # we dont have any files for this cfg/fold/split
            continue

        import src.models.Unets as modelmanager
        # create a model
        logging.info('Create model')
        model = modelmanager.create_unet(config)
        # load model weights
        model.load_weights(os.path.join(config['MODEL_PATH'], 'model.h5'))
        logging.info('loaded model weights as h5 file')


        # adjust config - no augment, no shuffle etc.
        # create a list of generators
        pred_config = config.copy()
        pred_config['SHUFFLE'] = False
        pred_config['AUGMENT'] = False
        pred_config['BATCHSIZE'] = 1
        pred_config['HIST_MATCHING'] = False
        # While inference we dont have a mask file, maske sure the generator works this way.
        gens = sliceable(DataGenerator ,x=files_filtered ,y=None, config=pred_config)

        # loop over filtered files, predict and write pred file to export_path_full
        from src.data.Postprocess import clean_3d_prediction_2d_cc
        for i in range(len(files_filtered)):

            # select next 4D CMR file, for patient id extraction
            temp_orig_f = files_filtered[i]
            orig_sitk = sitk.ReadImage(temp_orig_f)
            describe_sitk(orig_sitk)

            # select the next generator and predict
            gen = gens[i]
            pred = model.predict(gen)

            # create flat arrays based on the thresholdes channels/labels
            pred_flat = np.zeros((pred.shape[:-1]))
            pred_flat[pred[..., 0] > 0.5] = 1
            pred_flat[pred[..., 1] > 0.5] = 2
            pred_flat[pred[..., 2] > 0.5] = 3

            # reshape, create 4D numpy t,z,y,x
            pred_flat = pred_flat.reshape((*sitk.GetArrayFromImage(orig_sitk).shape[:2] ,*pred_flat.shape[1:]))
            pred_flat = pred_flat.astype(np.uint8)

            # minor postprocessing with connected component filtering
            pred_flat = np.stack([clean_3d_prediction_2d_cc(nda3d) for nda3d in pred_flat])

            # create 4D sitk x,y,z,t
            pred_sitk = sitk.JoinSeries([sitk.GetImageFromArray(pred_3d_flat) for pred_3d_flat in pred_flat])
            print(pred_sitk.GetSize())
            exp_spacing = tuple(reversed(pred_config.get('SPACING')))
            exp_spacing = (*exp_spacing, orig_sitk.GetSpacing()[2] ,1)
            print(exp_spacing)
            pred_sitk.SetSpacing(exp_spacing)

            pred_path = os.path.join(config.get('EXP_PATH'), export_suffix)
            ensure_dir(pred_path)
            sitk.WriteImage(pred_sitk, os.path.join(pred_path, '{}_msk.nrrd'.format
                (os.path.basename(files_filtered[i]).split('.')[0])))

    return pred_path