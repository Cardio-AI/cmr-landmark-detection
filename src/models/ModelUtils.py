import logging
import os
import tensorflow as tf
from tensorflow import keras


def load_pretrained_model(config=None, metrics=None, comp=True, multigpu=False, custom_objects={}):
    """
    Load a pre-trained keras model
    for a given model.json file and the weights as h5 file

    :param config: dict
    :param metrics: keras or tensorflow loss function in a list
    :param comp: bool, compile the model or not
    :multigpu: wrap model in multi gpu wrapper
    :param tf2model: bool if this model was saved with the new keras api model.save, with a folder and the full graph
    :return: tf.keras.model
    """
    import traceback
    if config is None:
        config = {}
    if metrics is None:
        metrics = [keras.metrics.binary_accuracy]

    gpu_ids = config.get('GPU_IDS', '1').split(',')
    loss_f = config.get('LOSS_FUNCTION', keras.losses.categorical_crossentropy)
    model_path = config.get('MODEL_PATH', './')

    try:
        # use the tf 2.x api to load the full model
        logging.info('load model with keras api')
        model = tf.keras.models.load_model(model_path,  custom_objects=custom_objects,compile=False,)
    except Exception as e:
        logging.info(str(e))
        logging.debug(traceback.format_exc())
        # earlier models need to be loaded via json and weights file
        logging.info('Keras API failed, use json repr. load model from: {} .'.format(os.path.join(model_path, 'model.json')))
        json = open(os.path.join(model_path, 'model.json')).read()
        logging.info('loading model description')
        model = tf.keras.models.model_from_json(json)
        try:
            logging.info('loading model weights')
            import glob
            h5_file = sorted(glob.glob(os.path.join(model_path, '*.h5')))[0]
            model.load_weights(h5_file)
            # make sure to work with wrapped multi-gpu models, tensorflow < 2.x
            """if multigpu:
                logging.info('multi GPU model, try to unpack the model and load weights again')
                model = model.layers[-2]
                model = multi_gpu_model(model, gpus=len(gpu_ids), cpu_merge=False) if (len(gpu_ids) > 1) else model"""
        except Exception as e:
            # some models are wrapped two times into a keras multi-gpu model, so we need to unpack it - hack
            logging.info(str())
            tf_weights = os.path.join(model_path, '/variables') # use tf weight format
            model.load_weights(tf_weights)
            """logging.info(str(e))
            logging.info('multi GPU model, try to unpack the model and load weights again')
            model = model.layers[-2]
            model.load_weights(os.path.join(model_path, 'checkpoint.h5'))"""
            pass


        if comp:
            try:
                # try to compile with given params, else use fallback parameters
                model.compile(optimizer=get_optimizer(config), loss=loss_f, metrics=metrics)
                logging.info('model compiled')

            except Exception as e:
                logging.error('Failed to compile with given parameters, use default vaules: {}'.format(str(e)))
                model.compile(optimizer='adam', loss=loss_f, metrics=metrics)
        logging.info('model {} loaded'.format(os.path.join(model_path, 'model.json')))
    return model

def get_optimizer(config, name_suff=''):
    """
    Returns a tf.keras.optimizer
    default is an Adam optimizer
    :param config: Key, value dict, Keys in upper letters
    :name_suff: string name suffix for the optimizer (if we wrap different models and optimizers)
    :return: tf.keras.optimizer
    """

    opt = config.get('OPTIMIZER', 'Adam')
    lr = config.get('LEARNING_RATE', 0.001)
    ep = config.get('EPSILON', 1e-08)
    de = config.get('DECAY', 0.0)

    optimizer = None

    opt = opt.lower()

    if opt == 'adagrad':
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)
    elif opt == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
    elif opt == 'adadelta':
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=lr)
    elif opt == 'radam':
        # need to: pip install keras-rectified-adam
        try:
            from keras_radam import RAdam
            optimizer = RAdam()
        except Exception as e:
            logging.error(str(e), 'Did u install radam? --> pip install keras-rectified-adam')
    elif opt == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr=lr, name=opt+name_suff)
    elif opt == 'nadam':
        optimizer = tf.keras.optimizers.Nadam(lr=lr, name=opt + name_suff)
    elif opt == 'sgd':
        optimizer = tf.keras.optimizers.SGD(
            lr=lr, nesterov=True, name=opt+name_suff)
    else:
        # no optimizer defined, use the adam with standard parameters
        optimizer = tf.keras.optimizers.Adam()

    logging.debug('Optimizer: {}'.format(opt))
    return optimizer