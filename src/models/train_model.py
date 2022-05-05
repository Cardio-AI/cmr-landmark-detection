def train_fold(config, in_memory=True):
    # make sure all neccessary params in config are set
    # if not set them with default values
    from src.utils.Tensorflow_helper import choose_gpu_by_id
    # ------------------------------------------define GPU id/s to use
    GPU_IDS = config.get('GPU_IDS', '0,1')
    GPUS = choose_gpu_by_id(GPU_IDS)
    print(GPUS)
    # ------------------------------------------ import helpers
    # this should import glob, os, and many other standard libs
    import tensorflow as tf
    tf.get_logger().setLevel('FATAL')
    import gc, logging, os, datetime
    from logging import info

    # local imports
    from src.utils.Utils_io import Console_and_file_logger, init_config, ensure_dir
    from src.utils.KerasCallbacks import get_callbacks
    from src.data.Dataset import get_trainings_files
    from src.data.Generators import DataGenerator
    import src.models.Loss_and_metrics as metr
    import src.models.Unets as modelmanager

    # import external libs
    from time import time
    t0 = time()

    # make all config params known to the local namespace
    locals().update(config)

    EXPERIMENT = config.get('EXPERIMENT')
    FOLD = config.get('FOLD')

    EXPERIMENT = '{}f{}'.format(EXPERIMENT, FOLD)
    """timestemp = str(datetime.datetime.now().strftime(
        "%Y-%m-%d_%H_%M"))"""  # add a timestep to each project to make repeated experiments unique

    EXPERIMENTS_ROOT = 'exp/'
    EXP_PATH = config.get('EXP_PATH')
    FOLD_PATH = os.path.join(EXP_PATH, 'f{}'.format(FOLD))
    MODEL_PATH = os.path.join(FOLD_PATH, 'model', )
    TENSORBOARD_PATH = os.path.join(FOLD_PATH, 'tensorboard_logs')
    CONFIG_PATH = os.path.join(FOLD_PATH, 'config')

    ensure_dir(MODEL_PATH)
    ensure_dir(TENSORBOARD_PATH)
    ensure_dir(CONFIG_PATH)

    DATA_PATH_SAX = config.get('DATA_PATH_SAX')
    DF_FOLDS = config.get('DF_FOLDS')
    EPOCHS = config.get('EPOCHS', 100)

    # Check if these channels are given
    metrics = [
        metr.dice_coef_labels,  # combination channel
        metr.dice_coef_myo,  # former Myo
        metr.dice_coef_lv,  # former LV
        metr.dice_coef_rv # third channel, not needed
    ]

    Console_and_file_logger(path=EXP_PATH, log_lvl=logging.INFO)
    config = init_config(config=locals(), save=True)
    """logging.info('Is built with tensorflow: {}'.format(tf.test.is_built_with_cuda()))
    logging.info('Visible devices:\n{}'.format(tf.config.list_physical_devices()))
    logging.info('Local devices: \n {}'.format(device_lib.list_local_devices()))"""

    # Load SAX volumes
    x_train, y_train, x_val, y_val = get_trainings_files(data_path=DATA_PATH_SAX, path_to_folds_df=DF_FOLDS, fold=FOLD)
    logging.info('SAX train CMR: {}, SAX train masks: {}'.format(len(x_train), len(y_train)))
    logging.info('SAX val CMR: {}, SAX val masks: {}'.format(len(x_val), len(y_val)))

    # create a batch generator
    batch_generator = DataGenerator(x_train, y_train, config=config, in_memory=in_memory)
    val_config = config.copy()
    val_config['AUGMENT_GRID'] = False  # make sure no augmentation will be applied to the validation data
    val_config['AUGMENT'] = False
    val_config['HIST_MATCHING'] = False
    validation_generator = DataGenerator(x_val, y_val, config=val_config, in_memory=in_memory)
    info('Done!')

    # create new model
    logging.info('Create model')
    model = modelmanager.create_unet(config, metrics, supervision=False)
    model.summary()

    # write the model summary to a txt file
    with open(os.path.join(EXP_PATH, 'model_summary.txt'), 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    tf.keras.utils.plot_model(
        model, show_shapes=True,
        to_file=os.path.join(EXP_PATH, 'model.png'),
        show_layer_names=True,
        rankdir='TB',
        expand_nested=True,
        dpi=96
    )

    # training
    initial_epoch = 0
    cb = get_callbacks(config, batch_generator, validation_generator)
    print('start training')
    # EPOCHS = 1
    model.fit(
        x=batch_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=cb,
        initial_epoch=initial_epoch,
        max_queue_size=config.get('QUEUE_SIZE', 12),
        verbose=1)

    try:
        # free as much memory as possible
        del batch_generator
        del validation_generator
        del model
        del cb
        gc.collect()

        # here we should add the prediction and evaluation:
        from src.models.predict_model import pred_fold
        pred_fold(config)
        # exp_path = config.get('EXP_PATH')
        # evaluate(exp_path)

    except Exception as e:
        logging.error(e)

    logging.info('Fold {} finished after {:0.3f} sec'.format(FOLD, time() - t0))
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
    import src.models.Loss_and_metrics as metr
    # import external libs
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    import cv2

    EXPERIMENTS_ROOT = 'exp/'

    if args.cfg:
        import json
        cfg = args.cfg
        print('config given: {}'.format(cfg))
        # load the experiment config
        with open(cfg, encoding='utf-8') as data_file:
            config = json.loads(data_file.read())

        # if config given, define new paths, so that we make sure that:
        # 1. we dont overwrite a previous config
        # 2. we store the experiment in the current source directory (cluster/local)
        EXPERIMENT = config.get('EXPERIMENT', 'UNDEFINED')
        timestemp = str(datetime.datetime.now().strftime(
            "%Y-%m-%d_%H_%M"))  # ad a timestep to each project to make repeated experiments unique

        config['EXP_PATH'] = os.path.join(EXPERIMENTS_ROOT, EXPERIMENT, timestemp)

        config['MODEL_PATH'] = os.path.join(config['EXP_PATH'], 'model', )
        config['TENSORBOARD_PATH'] = os.path.join(config['EXP_PATH'], 'tensorboard_logs')
        config['CONFIG_PATH'] = os.path.join(config['EXP_PATH'], 'config')
        config['HISTORY_PATH'] = os.path.join(config['EXP_PATH'], 'history')
        # Console_and_file_logger(path=config['EXP_PATH'])
        # this could be more dynamic, This loss worked the best for the ventricle labels
        if 'BcdDiceLoss' in config.get('LOSS_FUNCTION', ''):
            config['LOSS_FUNCTION'] = metr.BceDiceLoss()
            #config['LOSS_FUNCTION'] = metr.bce_dice_loss

        else:
            # handle default - if no loss is specified
            config['LOSS_FUNCTION'] = tf.keras.losses.MSE()

        if args.data:  # if we specified a different data path (training from workspace or local node disk)
            config['DATA_PATH_SAX'] = os.path.join(args.data, "2D/")
            config['DF_FOLDS'] = os.path.join(args.data, "df_kfold.csv")
            config['DATA_PATH_ORIG'] = os.path.join(args.data, 'original')
        # we dont need to initialise this config, as it should already have the correct format,
        # The fold configs will be saved with each fold run
        # config = init_config(config=config, save=False)
        print(config)
    else:
        print('no config given, build a new one')
        return


    for f in config.get('FOLDS', [0]):
        print('starting fold: {}'.format(f))
        config_ = config.copy()
        config_['FOLD'] = f
        train_fold(config_, in_memory=True)
        print('training of fold: {} finished'.format(f))
    from src.models.evaluate_cv import evaluate_cv
    #evaluate_cv(os.path.join(EXPERIMENTS_ROOT, EXPERIMENT), args.data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='train a RV IP detection/segmentation model on CMR images')

    # usually these two parameters should encapsulate all experiment parameters
    parser.add_argument('-cfg', action='store', default=None,help='path to an experiment config, you can find examples in exp/template_cfgs')
    parser.add_argument('-data', action='store', default=None,help='path to the data-root folder, please check src/Dataset/make_data.py or notebooks/Dataset/prepare_dataipynb for further hints')
    parser.add_argument('-inmemory', action='store', default=None,help='Generator works inmemory (cluster-based training), needs to be checked')  # this generator cant handle inmemory so far

    results = parser.parse_args()
    print('given parameters: {}'.format(results))
    assert results.cfg != None, 'no config given'
    assert results.data != None, 'no data given'

    try:
        main(results)
    except Exception as e:
        print(e)
    exit()
