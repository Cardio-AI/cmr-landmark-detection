import logging
import os
import random

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as kl

import src
import src.models.KerasLayers as ownkl
import src.models.ModelUtils as mutils
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

# address some interface discrepancies when using tensorflow.keras
# hack to use the load_from_json in tf otherwise we get an exception
# adapted/modified hack from here:
# https://github.com/keras-team/keras-contrib/issues/488
if "slice" not in keras.backend.__dict__:
    # this is a good indicator that we are using tensorflow.keras
    print('using tensorflow, need to monkey patch')
    try:
        # at first try to monkey patch what we need

        try:
            tf.python.keras.backend.__dict__.update(
                is_tensor=tf.is_tensor,
                slice=tf.slice,
            )
        finally:
            print('tf.python.backend.slice overwritten by monkey patch')
    except Exception:
        print('monkey patch failed, override methods')
        # if that doesn't work we do a dirty copy of the code required
        import tensorflow as tf
        from tensorflow.python.framework import ops as tf_ops


        def is_tensor(x):
            return isinstance(x, tf_ops._TensorLike) or tf_ops.is_dense_tensor_like(x)


        def slice(x, start, size):
            x_shape = keras.int_shape(x)
            if (x_shape is not None) and (x_shape[0] is not None):
                len_start = keras.int_shape(start)[0] if is_tensor(start) else len(start)
                len_size = keras.int_shape(size)[0] if is_tensor(size) else len(size)
                if not (len(keras.int_shape(x)) == len_start == len_size):
                    raise ValueError('The dimension and the size of indices should match.')
            return tf.slice(x, start, size)


# Build U-Net model
def create_unet(config, metrics=None, networkname='unet', single_model=True, supervision=False):
    """
    Factory for a 2D/3D u-net for image segmentation
    :param config: Key value pairs for image size and other network parameters
    :param metrics: list of tensorflow or keras compatible metrics
    :param networkname: string, name of this model scope
    :param stacked_model: bool, if this model will be used in another model, applies softmax and compile or not
    :return: compiled tf.keras model
    """
    if tf.distribute.has_strategy():
        strategy = tf.distribute.get_strategy()
    else:
        # distribute the training with the mirrored data paradigm across multiple gpus if available, if not use gpu 0
        strategy = tf.distribute.MirroredStrategy(devices=config.get('GPUS', ["/gpu:0"]))
    with strategy.scope():

        inputs = Input((*config.get('DIM', [224, 224]), config.get('IMG_CHANNELS', 1)))
        print(inputs.shape)

        # define standard values according to convention over configuration paradigm
        metrics = [keras.metrics.binary_accuracy] if metrics is None else metrics
        activation = config.get('ACTIVATION', 'elu')
        loss_f = config.get('LOSS_FUNCTION', keras.losses.categorical_crossentropy) # standard implementation of Loss function is categorical cross entropy. Theoretically better suited for our project. 

        batch_norm = config.get('BATCH_NORMALISATION', False)
        use_upsample = config.get('USE_UPSAMPLE', 'False')
        pad = config.get('PAD', 'same')
        kernel_init = config.get('KERNEL_INIT', 'he_normal')
        mask_classes = config.get("MASK_CLASSES", 3)
        ndims = len(config.get('DIM', [10, 224, 224]))
        m_pool = config.get('M_POOL', (1, 2, 2))
        m_pool = m_pool[-ndims:]
        f_size = config.get('F_SIZE', (3, 3, 3))
        f_size = f_size[-ndims:]
        filters = config.get('FILTERS', 16)
        drop_1 = config.get('DROPOUT_MIN', 0.3)
        drop_3 = config.get('DROPOUT_MAX', 0.5)
        bn_first = config.get('BN_FIRST', False)

        depth = config.get('DEPTH', 4)
        Conv = getattr(kl, 'Conv{}D'.format(ndims))
        one_by_one = (1, 1, 1)[:ndims]

        # increase the dropout through the layer depth
        dropouts = list(np.linspace(drop_1, drop_3, depth))
        dropouts = [round(i, 1) for i in dropouts]

        outputs = unet(activation=activation,
                       batch_norm=batch_norm,
                       bn_first=bn_first,
                       depth=depth,
                       drop_3=drop_3,
                       dropouts=dropouts,
                       f_size=f_size,
                       filters=filters,
                       inputs=inputs,
                       kernel_init=kernel_init,
                       m_pool=m_pool,
                       ndims=ndims,
                       pad=pad,
                       use_upsample=use_upsample,
                       mask_classes=mask_classes,
                       supervision=supervision)


        # stacked models will be compiled later, dont return softmax or sigmoid outputs
        if single_model:
            outputs = Conv(mask_classes, one_by_one, activation='sigmoid', name='unet')(outputs) # WFT Data Science: 'sigmoid' ; output function, should be adapted to softmax for better performance and coupled with a Categorical Cross Entropy  + Dice as a loss function instead of the current BCE_Dice
            model = Model(inputs=[inputs], outputs=[outputs], name=networkname)
            model.compile(optimizer=mutils.get_optimizer(config, networkname), loss={'unet': loss_f}, metrics=metrics)
        else:
            model = Model(inputs=[inputs], outputs=[outputs], name=networkname)
        return model

# Build U-Net model
def create_unet_layer(config, metrics=None, networkname='unet', single_model=True, supervision=False):
    """
    Factory for a 2D/3D u-net for image segmentation
    :param config: Key value pairs for image size and other network parameters
    :param metrics: list of tensorflow or keras compatible metrics
    :param networkname: string, name of this model scope
    :param stacked_model: bool, if this model will be used in another model, applies softmax and compile or not
    :return: compiled tf.keras model
    """
    if tf.distribute.has_strategy():
        strategy = tf.distribute.get_strategy()
    else:
        # distribute the training with the mirrored data paradigm across multiple gpus if available, if not use gpu 0
        strategy = tf.distribute.MirroredStrategy(devices=config.get('GPUS', ["/gpu:0"]))
    with strategy.scope():

        inputs = Input((*config.get('DIM', [10, 224, 224]), config.get('IMG_CHANNELS', 1)))
        print(inputs.shape)

        # define standard values according to convention over configuration paradigm
        metrics = [keras.metrics.binary_accuracy] if metrics is None else metrics
        activation = config.get('ACTIVATION', 'elu')
        loss_f = config.get('LOSS_FUNCTION', keras.losses.categorical_crossentropy)
        batch_norm = config.get('BATCH_NORMALISATION', False)
        use_upsample = config.get('USE_UPSAMPLE', 'False')
        pad = config.get('PAD', 'same')
        kernel_init = config.get('KERNEL_INIT', 'he_normal')
        mask_classes = config.get("MASK_CLASSES", 4)
        m_pool = config.get('M_POOL', (1, 2, 2))
        f_size = config.get('F_SIZE', (3, 3, 3))
        filters = config.get('FILTERS', 16)
        drop_1 = config.get('DROPOUT_min', 0.3)
        drop_3 = config.get('DROPOUT_max', 0.5)
        bn_first = config.get('BN_FIRST', False)
        ndims = len(config.get('DIM', [10, 224, 224]))
        depth = config.get('DEPTH', 4)

        # calc filter size of the lower layers
        filters_decoder = filters
        for _ in range(depth):
             filters_decoder *= 2

        # increase the dropout through the layer depth

        dropouts = list(np.linspace(drop_1, drop_3, depth))
        dropouts = [round(i, 1) for i in dropouts]

        Conv = getattr(kl, 'Conv{}D'.format(ndims))
        one_by_one = (1, 1, 1)[-ndims:]

        x = ownkl.ConvEncoder(activation=activation, batch_norm=batch_norm, bn_first=bn_first,
                                     depth=depth, drop_3=drop_3, dropouts=dropouts, f_size=f_size,
                                     filters=filters, kernel_init=kernel_init, m_pool=m_pool,
                                     ndims=ndims, pad=pad)(inputs)

        x = ownkl.ConvDecoder(activation=activation, batch_norm=batch_norm, bn_first=bn_first,
                                     depth=depth, drop_3=drop_3, dropouts=dropouts, f_size=f_size,
                                     filters=filters_decoder, kernel_init=kernel_init, up_size=m_pool,
                                     ndims=ndims, pad=pad,use_upsample=use_upsample)(x)


        x = Conv(mask_classes, one_by_one, padding=pad, kernel_initializer=kernel_init,
                                    activation='sigmoid')(x)

        model = Model(inputs=[inputs], outputs=[x], name=networkname)

        # compile only single unets, stacked models will be compiled later
        print(loss_f)
        if single_model:
            model.compile(optimizer=mutils.get_optimizer(config, networkname),
                          loss=loss_f,
                          metrics=metrics)

    return model


# Build U-Net model
def create_unet_class(config, metrics=None, networkname='unet', single_model=True, supervision=False):
    """
    Factory for a 2D/3D u-net for image segmentation
    :param config: Key value pairs for image size and other network parameters
    :param metrics: list of tensorflow or keras compatible metrics
    :param networkname: string, name of this model scope
    :param stacked_model: bool, if this model will be used in another model, applies softmax and compile or not
    :return: compiled tf.keras model
    """
    if tf.distribute.has_strategy():
        strategy = tf.distribute.get_strategy()
    else:
        # distribute the training with the mirrored data paradigm across multiple gpus if available, if not use gpu 0
        strategy = tf.distribute.MirroredStrategy(devices=config.get('GPUS', ["/gpu:0"]))
    with strategy.scope():
        inputs_shape = config.get('DIM', [10, 224, 224])
        print(inputs_shape)
        inputs = Input(inputs_shape)
        print(inputs.shape)

        # define standard values according to convention over configuration paradigm
        metrics = [keras.metrics.binary_accuracy] if metrics is None else metrics
        activation = config.get('ACTIVATION', 'elu')
        loss_f = config.get('LOSS_FUNCTION', keras.losses.categorical_crossentropy)
        batch_norm = config.get('BATCH_NORMALISATION', False)
        use_upsample = config.get('USE_UPSAMPLE', 'False')
        pad = config.get('PAD', 'same')
        kernel_init = config.get('KERNEL_INIT', 'he_normal')
        mask_classes = config.get("MASK_CLASSES", 4)
        m_pool = config.get('M_POOL', (1, 2, 2))
        f_size = config.get('F_SIZE', (3, 3, 3))
        filters = config.get('FILTERS', 16)
        drop_1 = config.get('DROPOUT_min', 0.3)
        drop_3 = config.get('DROPOUT_max', 0.5)
        bn_first = config.get('BN_FIRST', False)
        ndims = len(config.get('DIM', [10, 224, 224]))
        depth = config.get('DEPTH', 4)

        # calc filter size of the lower layers
        filters_decoder = filters
        for l in range(depth):
             filters_decoder *= 2

        # increase the dropout through the layer depth

        dropouts = list(np.linspace(drop_1, drop_3, depth))
        dropouts = [round(i, 1) for i in dropouts]

        Conv = getattr(kl, 'Conv{}D'.format(ndims))
        one_by_one = (1, 1, 1)[-ndims:]

        model = Unet(dim=inputs_shape,
                     activation=activation,
                     batch_norm=batch_norm,
                     bn_first=bn_first,
                     depth=depth,
                     drop_3=drop_3,
                     dropouts=dropouts,
                     f_size=f_size,
                     filters=filters,
                     kernel_init=kernel_init,
                     m_pool=m_pool,
                     ndims=ndims,
                     pad=pad,
                     use_upsample=use_upsample,
                     mask_classes=mask_classes,
                     supervision=supervision)
        #model = Unet()
        model.build(input_shape=inputs_shape)

        # compile only single unets, stacked models will be compiled later
        if single_model:
            model.compile(optimizer=mutils.get_optimizer(config, networkname), loss={'unet': loss_f}, metrics=metrics)

    return model

def create_3d_wrapper_for_2d_unet_followed_3d_unet(config, metrics=None, supervision=False, unet_2d=None):
    """
    Create a stacked u-net, inject a 2D u-net or train a new one
    forward the masks from the 2D unet trough a 3D u-net
    :param config:
    :param metrics:
    :param unet_2d:
    :return:
    """

    metrics = [keras.metrics.binary_accuracy] if metrics is None else metrics
    activation = config.get('ACTIVATION', 'elu')
    loss_f = config.get('LOSS_FUNCTION', keras.losses.categorical_crossentropy)
    inputs = Input((*config.get('DIM', [224, 224]), config.get('IMG_CHANNELS', 1)))
    Conv = getattr(kl, 'Conv{}D'.format(3))
    one_by_one = (1, 1, 1)
    mask_classes = config.get("MASK_CLASSES", 4)

    if unet_2d:
        logging.info('use pre-trained 2d unet')
        unet_2d.trainable = False
    else:  # no 2D model injected, modify the config and create a new 2D-unet
        # modify the network parameters, that control the dimensionality of the graph
        logging.info('create a new 2D unet')
        config_2d = config.copy()
        config_2d['DIM'] = config.get('DIM', [10, 224, 224])[1:]
        config_2d['F_SIZE'] = config.get('F_SIZE', [3, 3, 3])[1:]
        config_2d['M_POOL'] = config.get('M_POOL', [1, 2, 2])[1:]
        unet_2d = create_unet(config=config_2d, metrics=metrics, networkname='2D-unet', single_model=True,
                              supervision=supervision)

    # shuffle 2d slices, to improve the model robustness
    slices = tf.unstack(inputs, axis=1)
    # shuffle the zipped 2D slices and the indices, to enable sorted stacking afterwards
    indicies = list(tf.range(inputs.shape[1]))
    zipped = list(zip(slices, indicies))
    random.shuffle(zipped)
    slices, indicies = zip(*zipped)
    # forward the shuffled 2D slices
    result = [unet_2d(s) for s in slices]
    # sort by indices, to stack in correct order
    result, _ = zip(*sorted(zip(result, indicies), key=lambda tup: tup[1]))
    output2d = K.stack(result, axis=1)

    config['IMG_CHANNELS'] = 4
    # Feed the segmentations from the 2D unet into the 3D Unet
    unet_3d = create_unet(config=config, metrics=metrics, networkname='3D-unet_followed_2D', single_model=False,
                          supervision=supervision)
    output3d = unet_3d(output2d)
    output3d = Conv(mask_classes, one_by_one, activation='softmax', dtype='float32')(output3d)

    model = Model(inputs=[inputs], outputs=[output3d], name='pretrained-2D-trainable-3D')
    model.compile(optimizer=mutils.get_optimizer(config, name_suff='pretrained-2D-trainable-3D'), loss=loss_f,
                  metrics=metrics)

    return model


def create_3d_wrapper_for_2d_unet(config, metrics=None, supervision=False, unet_2d=None, compile_=True):
    """
    Create a stacked u-net, inject a 2D u-net or train a new one (training will be not as good as training a real 2D unet)
    :param config:
    :param metrics:
    :param unet_2d:
    :return:
    """
    if tf.distribute.has_strategy():
        strategy = tf.distribute.get_strategy()
    else:
        print('create a new strategy')
        # distribute the training with the mirrored data paradigm across multiple gpus if available, if not use gpu 0
        strategy = tf.distribute.MirroredStrategy(devices=config.get('GPUS', ["/gpu:0"]))
    tf.print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
        metrics = [keras.metrics.binary_accuracy] if metrics is None else metrics
        activation = config.get('ACTIVATION', 'elu')
        loss_f = config.get('LOSS_FUNCTION', keras.losses.categorical_crossentropy)
        inputs = Input((*config.get('DIM', [224, 224]), config.get('IMG_CHANNELS', 1)))
        Conv = getattr(kl, 'Conv{}D'.format(3))
        one_by_one = (1, 1, 1)
        mask_classes = config.get("MASK_CLASSES", 4)

        if unet_2d:
            logging.info('use pre-trained 2d unet')
            unet_2d.trainable = False
        else:  # no 2D model injected, modify the config and create a new 2D-unet
            # modify the network parameters, that control the dimensionality of the graph
            logging.info('create a new 2D unet')
            config_2d = config.copy()
            config_2d['DIM'] = config.get('DIM', [10, 224, 224])[1:]
            config_2d['F_SIZE'] = config.get('F_SIZE', [3, 3, 3])[1:]
            config_2d['M_POOL'] = config.get('M_POOL', [1, 2, 2])[1:]
            unet_2d = create_unet(config=config_2d, metrics=metrics, networkname='2D-unet', single_model=True,
                                  supervision=supervision)

        # shuffle 2d slices, to improve the model robustness
        slices = tf.unstack(inputs, axis=1)
        # shuffle the zipped 2D slices and the indices, to enable sorted stacking afterwards
        indicies = list(tf.range(inputs.shape[1]))
        zipped = list(zip(slices, indicies))
        random.shuffle(zipped)
        slices, indicies = zip(*zipped)
        # forward the shuffled 2D slices
        result = [unet_2d(s) for s in slices]
        # sort by indices, to stack in correct order
        result, _ = zip(*sorted(zip(result, indicies), key=lambda tup: tup[1]))
        output2d = K.stack(result, axis=1)

        # output3d = Conv(mask_classes, one_by_one, activation='softmax', dtype='float32')(output2d)

        model = Model(inputs=[inputs], outputs=[output2d], name='pretrained-2D-wrapper')
        if True:
            model.compile(optimizer=mutils.get_optimizer(config, name_suff='pretrained-2D-wrapper'), loss=loss_f,
                          metrics=metrics)

        return model


def create_3d_wrapper_for_2d_unet_concat_input_followed_3d_unet(config, metrics=None, supervision=False, unet_2d=None):
    """
    Create a stacked u-net, inject a 2D u-net or train a new one
    concat the output (masks) with the input image and forward it together trough a 3D u-net
    :param config:
    :param metrics:
    :param unet_2d:
    :return:
    """

    metrics = [keras.metrics.binary_accuracy] if metrics is None else metrics
    activation = config.get('ACTIVATION', 'elu')
    loss_f = config.get('LOSS_FUNCTION', keras.losses.categorical_crossentropy)
    inputs = Input((*config.get('DIM', [224, 224]), config.get('IMG_CHANNELS', 1)))
    Conv = getattr(kl, 'Conv{}D'.format(3))
    one_by_one = (1, 1, 1)
    mask_classes = config.get("MASK_CLASSES", 4)

    if unet_2d:
        logging.info('use pre-trained 2d unet')
        unet_2d.trainable = False
    else:  # no 2D model injected, modify the config and create a new 2D-unet
        # modify the network parameters, that control the dimensionality of the graph
        logging.info('create a new 2D unet')
        config_2d = config.copy()
        config_2d['DIM'] = config.get('DIM', [10, 224, 224])[1:]
        config_2d['F_SIZE'] = config.get('F_SIZE', [3, 3, 3])[1:]
        config_2d['M_POOL'] = config.get('M_POOL', [1, 2, 2])[1:]
        unet_2d = create_unet(config=config_2d, metrics=metrics, networkname='2D-unet', single_model=True,
                              supervision=supervision)

    # shuffle 2d slices, to improve the model robustness
    slices = tf.unstack(inputs, axis=1)
    # shuffle the zipped 2D slices and the indices, to enable sorted stacking afterwards
    indicies = list(tf.range(inputs.shape[1]))
    zipped = list(zip(slices, indicies))
    random.shuffle(zipped)
    slices, indicies = zip(*zipped)
    # forward the shuffled 2D slices
    result = [unet_2d(s) for s in slices]
    # sort by indices, to stack in correct order
    result, _ = zip(*sorted(zip(result, indicies), key=lambda tup: tup[1]))
    output2d = K.stack(result, axis=1)
    combined_output = tf.keras.layers.concatenate([output2d, inputs], axis=-1)
    # output2d has 4 channels = 1 channel from the input MRI
    config['IMG_CHANNELS'] = 5
    # Feed the segmentations from the 2D u-net into the 3D Unet, stack it with the predictions of the 2D u-net
    unet_3d = create_unet(config=config, metrics=metrics, networkname='3D-unet_followed_2D', single_model=False,
                          supervision=supervision)
    output3d = unet_3d(combined_output)
    output3d = Conv(mask_classes, one_by_one, activation='softmax', dtype='float32')(output3d)

    model = Model(inputs=[inputs], outputs=[output3d], name='pretrained-2D-stacked-trainable-3D')
    model.compile(optimizer=mutils.get_optimizer(config, name_suff='pretrained-2D-stacked-trainable-3D'), loss=loss_f,
                  metrics=metrics)

    return model


def create_3d_wrapper_for_2d_unet_avg_with_3D_unet(config, metrics=None, supervision=False, unet_2d=None):
    """
    create a 2D/3D u-net for image segmentation
    This model uses a pre-trained (fixed weights) 2D unet and combine the predictions with a trainable 3D unet
    :param config: Key value pairs for image size and other network parameters
    :param metrics: list of tensorflow or keras compatible metrics
    :returns compiled keras model
    """

    # define standard values according to convention over configuration paradigm
    metrics = [keras.metrics.binary_accuracy] if metrics is None else metrics
    activation = config.get('ACTIVATION', 'elu')
    loss_f = config.get('LOSS_FUNCTION', keras.losses.categorical_crossentropy)
    pad = config.get('PAD', 'same')
    kernel_init = config.get('KERNEL_INIT', 'he_normal')
    mask_classes = config.get("MASK_CLASSES", 4)

    # define the final conv layer
    Conv = getattr(kl, 'Conv{}D'.format(3))
    one_by_one = (1, 1, 1)

    # distribute the training with the mirrored data paradigm across multiple gpus if available, if not use gpu 0
    # strategy for multi-GPU usage not necessary for single GPU usage, But should work

    if tf.distribute.has_strategy():
        strategy = tf.distribute.get_strategy()
    else:
        # distribute the training with the mirrored data paradigm across multiple gpus if available, if not use gpu 0
        strategy = tf.distribute.MirroredStrategy(devices=config.get('GPUS', ["/gpu:0"]))
    tf.print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():

        inputs = Input((*config.get('DIM', [224, 224]), config.get('IMG_CHANNELS', 1)))

        # modify the network parameters, that control the dimensionality of the graph
        config_2d = config.copy()
        config_2d['DIM'] = config.get('DIM', [10, 224, 224])[1:]
        config_2d['F_SIZE'] = config.get('F_SIZE', [3, 3, 3])[1:]
        config_2d['M_POOL'] = config.get('M_POOL', [1, 2, 2])[1:]

        # by creating a submodel all weights will be shared
        if unet_2d:
            logging.info('use the pre-trained 2D unet')
            # use the pre-trained unet_2d, freeze the weights
            # pre-trained 2D u-nets perform better than 2D unets wrapped in a 3D unet (this is related to the trainings process)
            unet_2d.trainable = False
        else:  # create a new trainable unet if non is given
            config_2d = config.copy()
            config_2d['DIM'] = config.get('DIM', [10, 224, 224])[1:]
            config_2d['F_SIZE'] = config.get('F_SIZE', [3, 3, 3])[1:]
            config_2d['M_POOL'] = config.get('M_POOL', [1, 2, 2])[1:]
            logging.info('no pre-trained model given, create a new one')
            unet_2d = create_unet(config=config_2d, metrics=metrics, networkname='2D-unet', single_model=False,
                                  supervision=supervision)

        # shuffle 2d slices, to improve the model robustness
        slices = tf.unstack(inputs, axis=1)
        indicies = list(tf.range(inputs.shape[1]))
        zipped = list(zip(slices, indicies))
        random.shuffle(zipped)
        slices, indicies = zip(*zipped)

        # forward the shuffled 2D slices
        result = [unet_2d(s) for s in slices]
        # sort by indices, to stack in correct order
        result, _ = zip(*sorted(zip(result, indicies), key=lambda tup: tup[1]))
        output2d = K.stack(result, axis=1)

        # create a 3D unet
        unet_3d = create_unet(config=config, metrics=metrics, networkname='3D-unet', single_model=False,
                              supervision=supervision)
        output3d = unet_3d(inputs)
        output3d = Conv(mask_classes, one_by_one, activation='softmax', dtype='float32')(output3d)

        # concatenate the stacked output of the 2D unet and the output of the 3D unet along the last axis
        # outputs = concatenate([output3d, output2d], axis=-1)
        # weight_2d = tf.Variable(1., trainable=True)
        # weight_3d = tf.Variable(1., trainable=True)
        outputs = tf.keras.layers.average([output2d, output3d])
        # outputs = tf.keras.layers.average([tf.math.scalar_mul(weight_3d, output3d), tf.math.scalar_mul(weight_2d, output2d)])

        # weight the two predicted segmentation volumes over the surrounding probabilities
        # outputs = Conv(512, (3,3,3), kernel_initializer = kernel_init, padding = pad, activation=activation)(outputs)
        # reduce the channels to the desired number of labels
        outputs = Conv(mask_classes, one_by_one, kernel_initializer=kernel_init, padding=pad, activation='softmax',
                       dtype='float32')(outputs)

        # reduce the channels to the desired number of labels
        # outputs = Conv(mask_classes, (3,3,3), kernel_initializer=kernel_init, padding = pad, activation='softmax', dtype='float32')(outputs)

        model = Model(inputs=[inputs], outputs=[outputs], name='pre-trained_2D_and_trainable_3D_avg')
        model.compile(optimizer=mutils.get_optimizer(config), loss=loss_f, metrics=metrics)

    return model


# build a combined (2D & 3D) Unet with shared weights for the 2D U-net
def create_2d_3d_avg_model(config, metrics=None, supervision=False):
    """
    create a 2D/3D u-net for image segmentation
    :param config: Key value pairs for image size and other network parameters
    :param metrics: list of tensorflow or keras compatible metrics
    :returns compiled keras model
    """

    # define standard values according to convention over configuration paradigm
    metrics = [keras.metrics.binary_accuracy] if metrics is None else metrics
    activation = config.get('ACTIVATION', 'elu')
    loss_f = config.get('LOSS_FUNCTION', keras.losses.categorical_crossentropy)
    pad = config.get('PAD', 'same')
    kernel_init = config.get('KERNEL_INIT', 'he_normal')
    mask_classes = config.get("MASK_CLASSES", 4)

    # define the final conv layer
    Conv = getattr(kl, 'Conv{}D'.format(3))
    one_by_one = (1, 1, 1)

    # distribute the training with the mirrored data paradigm across multiple gpus if available, if not use gpu 0
    # strategy for multi-GPU usage not necessary for single GPU usage, But should work

    if tf.distribute.has_strategy():
        strategy = tf.distribute.get_strategy()
    else:
        # distribute the training with the mirrored data paradigm across multiple gpus if available, if not use gpu 0
        strategy = tf.distribute.MirroredStrategy(devices=config.get('GPUS', ["/gpu:0"]))
    tf.print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():

        inputs = Input((*config.get('DIM', [224, 224]), config.get('IMG_CHANNELS', 1)))

        # modify the network parameters, that control the dimensionality of the graph
        config_2d = config.copy()
        config_2d['DIM'] = config.get('DIM', [10, 224, 224])[1:]
        config_2d['F_SIZE'] = config.get('F_SIZE', [3, 3, 3])[1:]
        config_2d['M_POOL'] = config.get('M_POOL', [1, 2, 2])[1:]

        # by creating a submodel all weights will be shared
        unet_2d = create_unet(config=config_2d, metrics=metrics, networkname='2D-unet', single_model=False,
                              supervision=supervision)

        # shuffle 2d slices, to improve the model robustness
        slices = tf.unstack(inputs, axis=1)
        indicies = list(tf.range(inputs.shape[1]))
        zipped = list(zip(slices, indicies))
        random.shuffle(zipped)
        slices, indicies = zip(*zipped)

        # forward the shuffled 2D slices
        result = [unet_2d(s) for s in slices]
        # sort by indices, to stack in correct order
        result, _ = zip(*sorted(zip(result, indicies), key=lambda tup: tup[1]))
        output2d = K.stack(result, axis=1)

        output2d = Conv(mask_classes, one_by_one, activation='softmax', dtype='float32')(output2d)

        # create a 3D unet
        unet_3d = create_unet(config=config, metrics=metrics, networkname='3D-unet', single_model=False,
                              supervision=supervision)
        output3d = unet_3d(inputs)
        output3d = Conv(mask_classes, one_by_one, activation='softmax', dtype='float32')(output3d)

        # concatenate the stacked output of the 2D unet and the output of the 3D unet along the last axis
        # outputs = concatenate([output3d, output2d], axis=-1)
        # weight_2d = tf.Variable(1., trainable=True)
        # weight_3d = tf.Variable(1., trainable=True)
        outputs = tf.keras.layers.average([output2d, output3d])
        # outputs = tf.keras.layers.average([tf.math.scalar_mul(weight_3d, output3d), tf.math.scalar_mul(weight_2d, output2d)])

        # weight the two predicted segmentation volumes over the surrounding probabilities
        # outputs = Conv(16, (3,3,3), kernel_initializer = kernel_init, padding = pad, activation=activation)(outputs)
        # reduce the channels to the desired number of labels
        # outputs = Conv(mask_classes, one_by_one, kernel_initializer=kernel_init, padding=pad, activation='softmax', dtype='float32')(outputs)

        # reduce the channels to the desired number of labels
        # outputs = Conv(mask_classes, (3,3,3), kernel_initializer=kernel_init, padding = pad, activation='softmax', dtype='float32')(outputs)

        model = Model(inputs=[inputs], outputs=[outputs], name='stacked-2D-unet')
        model.compile(optimizer=mutils.get_optimizer(config), loss=loss_f, metrics=metrics)

    return model


class Unet(tf.keras.Model):

    def __init__(self, dim=[10,224,224], activation='elu', batch_norm=False, bn_first=False, depth=4, drop_3=0.5,
                 dropouts=[0.2, 0.3, 0.4, 0.5], f_size=(3, 3, 3), filters=16,
                 kernel_init='he_normal', m_pool=(1, 2, 2), ndims=3, pad='same', use_upsample=True,
                 mask_classes=4, supervision=False):

        """
        Unet model as tf.keras subclass
        :param dim:
        :param activation:
        :param batch_norm:
        :param bn_first:
        :param depth:
        :param drop_3:
        :param dropouts:
        :param f_size:
        :param filters:
        :param kernel_init:
        :param m_pool:
        :param ndims:
        :param pad:
        :param use_upsample:
        :param mask_classes:
        :param supervision:
        """


        super(self.__class__, self).__init__()

        self.dim = dim
        self.use_upsample = use_upsample
        self.pad = pad
        self.ndims = ndims
        self.m_pool = m_pool
        self.kernel_init = kernel_init
        self.filters = filters
        # calc filter size of the lower layers
        for l in range(depth):
            filters *= 2
        self.filters_decoder = filters
        self.f_size = f_size
        self.dropouts = dropouts
        self.drop_3 = drop_3
        self.depth = depth
        self.bn_first = bn_first
        self.batch_norm = batch_norm
        self.activation = activation
        self.mask_classes = mask_classes
        self.supervision = supervision

        Conv = getattr(kl, 'Conv{}D'.format(self.ndims))
        one_by_one = (1, 1, 1)[-self.ndims:]

        self.inputs = Input(shape=(*self.dim, 1))

        self.enc = ownkl.ConvEncoder(activation=self.activation,batch_norm=self.batch_norm,bn_first=self.bn_first,
                                     depth=self.depth,drop_3=self.drop_3,dropouts=self.dropouts,f_size=self.f_size,
                                     filters=self.filters,kernel_init=self.kernel_init,m_pool=self.m_pool,ndims=self.ndims,pad=self.pad)
        self.dec = ownkl.ConvDecoder(activation=self.activation, batch_norm=self.batch_norm, bn_first=self.bn_first,
                                     depth=self.depth, drop_3=self.drop_3, dropouts=self.dropouts, f_size=self.f_size,
                                     filters=self.filters_decoder, kernel_init=self.kernel_init, up_size=self.m_pool,
                                     ndims=self.ndims, pad=self.pad, use_upsample=use_upsample)

        self.conv1 = Conv(filters=self.filters, kernel_size=f_size, kernel_initializer=self.kernel_init,
                         padding=self.pad)
        self.drop1 = Dropout(drop_3)
        self.conv2 = Conv(filters=self.filters, kernel_size=f_size, kernel_initializer=self.kernel_init,
                         padding=self.pad)
        self.conv_block = ownkl.ConvBlock(filters=self.filters_decoder,f_size=f_size,activation=activation,batch_norm=batch_norm,kernel_init=kernel_init,pad=pad,bn_first=bn_first,ndims=ndims)

        self.conv_one_by_one = Conv(mask_classes, one_by_one, padding=pad, kernel_initializer=kernel_init, activation=activation)

        self.out = self.call(self.inputs)


    def call(self, x, training=None, mask=None):
        """
        Create a U-net based graph
        :param x: tf.tensor
        :param training: bool
        :param mask: bool
        :return:
        """

        enc, skips = self.enc(x)
        #enc = self.conv1(enc)
        #enc = self.drop1(enc)
        #enc = self.conv2(enc)
        #enc = self.conv_block(enc)

        dec = self.dec([enc,skips])
        x = self.conv_one_by_one(dec)
        return x


    def summary(self):
        """
        Hack, overwrite the summary function to work with subclassed models
        This creates a model with a concrete output shape
        and returns model().summary()
        :return:
        """
        return Model(inputs=[self.inputs], outputs=self.call(self.inputs)).summary()



def unet(activation, batch_norm, bn_first, depth, drop_3, dropouts, f_size, filters, inputs,
         kernel_init, m_pool, ndims, pad, use_upsample, mask_classes, supervision=False):
    """
    unet 2d or 3d for the functional tf.keras api
    :param activation:
    :param batch_norm:
    :param bn_first:
    :param depth:
    :param drop_3:
    :param dropouts:
    :param f_size:
    :param filters:
    :param inputs:
    :param kernel_init:
    :param m_pool:
    :param ndims:
    :param pad:
    :param use_upsample:
    :param mask_classes:
    :return:
    """

    filters_init = filters
    encoder = list()
    decoder = list()
    dropouts = dropouts.copy()

    Conv = getattr(kl, 'Conv{}D'.format(ndims))
    one_by_one = (1, 1, 1)[-ndims:]

    # build the encoder
    for l in range(depth):

        if len(encoder) == 0:
            # first block
            input_tensor = inputs
        else:
            # all other blocks, use the max-pooled output of the previous encoder block
            # remember the max-pooled output from the previous layer
            input_tensor = encoder[-1][1]
        encoder.append(
            ownkl.downsampling_block_fn(inputs=input_tensor,
                                     filters=filters,
                                     f_size=f_size,
                                     activation=activation,
                                     drop=dropouts[l],
                                     batch_norm=batch_norm,
                                     kernel_init=kernel_init,
                                     pad=pad,
                                     m_pool=m_pool,
                                     bn_first=bn_first,
                                     ndims=ndims))
        filters *= 2
    # middle part
    input_tensor = encoder[-1][1]
    fully = ownkl.conv_layer_fn(inputs=input_tensor, filters=filters, f_size=f_size,
                             activation=activation, batch_norm=batch_norm, kernel_init=kernel_init,
                             pad=pad, bn_first=bn_first, ndims=ndims)
    fully = Dropout(drop_3)(fully)
    fully = ownkl.conv_layer_fn(inputs=fully, filters=filters, f_size=f_size,
                             activation=activation, batch_norm=batch_norm, kernel_init=kernel_init,
                             pad=pad, bn_first=bn_first, ndims=ndims)
    # build the decoder
    decoder.append(fully)
    for l in range(depth):
        # take the output of the previous decoder block and the output of the corresponding
        # encoder block
        input_lower = decoder[-1]
        input_skip = encoder.pop()[0]
        filters //= 2
        decoder.append(
            ownkl.upsampling_block_fn(lower_input=input_lower,
                                   conv_input=input_skip,
                                   use_upsample=use_upsample,
                                   filters=filters,
                                   f_size=f_size,
                                   activation=activation,
                                   drop=dropouts.pop(),
                                   batch_norm=batch_norm,
                                   up_size=m_pool,
                                   bn_first=bn_first,
                                   ndims=ndims))

    # deep supervision
    # current test
    if supervision:
        try:
            UpSampling = getattr(ownkl, 'UpSampling{}DInterpol'.format(ndims))
        except Exception as e:
            logging.info('own Upsampling layer not available, fallback to tensorflow upsampling')
            UpSampling = getattr(kl, 'UpSampling{}D'.format(ndims))

        # mask from the pre-last upsampling block
        lower_mask = decoder[-2]
        lower_mask = Conv(filters_init, one_by_one, padding=pad, kernel_initializer=kernel_init, activation=activation)(
            lower_mask)

        ### current test run, transpose vs upsampling, upsampling is better, maybe because the other upsampling is also done with u
        # ConvTranspose = getattr(kl, 'Conv{}DTranspose'.format(ndims))
        # lower_mask = ConvTranspose(filters=mask_classes, kernel_size=f_size, strides=m_pool, padding=pad, kernel_initializer=kernel_init,
        #                        activation=activation)(lower_mask)

        lower_mask = UpSampling(size=m_pool)(lower_mask)

    outputs = decoder[-1]

    if supervision:
        # outputs = concatenate([lower_mask, outputs])
        outputs = Multiply()([lower_mask, outputs])

        # use either a 3x3 conv or 1x1
        # 16 or 4 filters
        # outputs = Conv(mask_classes, f_size, padding=pad, kernel_initializer=kernel_init, activation=activation)(outputs)

    return outputs


def unet_save(activation, batch_norm, bn_first, depth, drop_3, dropouts, f_size, filters, inputs,
              kernel_init, m_pool, ndims, pad, use_upsample, mask_classes, supervision=False):
    """
    unet 2d or 3d for the functional tf.keras api
    :param activation:
    :param batch_norm:
    :param bn_first:
    :param depth:
    :param drop_3:
    :param dropouts:
    :param f_size:
    :param filters:
    :param inputs:
    :param kernel_init:
    :param m_pool:
    :param ndims:
    :param pad:
    :param use_upsample:
    :param mask_classes:
    :return:
    """

    filters_init = filters
    encoder = list()
    decoder = list()
    dropouts = dropouts.copy()

    Conv = getattr(kl, 'Conv{}D'.format(ndims))
    one_by_one = (1, 1, 1)[-ndims:]

    # build the encoder
    for l in range(depth):

        if len(encoder) == 0:
            # first block
            input_tensor = inputs
        else:
            # all other blocks, use the max-pooled output of the previous encoder block
            # remember the max-pooled output from the previous layer
            input_tensor = encoder[-1][1]
        encoder.append(
            ownkl.downsampling_block(inputs=input_tensor,
                                     filters=filters,
                                     f_size=f_size,
                                     activation=activation,
                                     drop=dropouts[l],
                                     batch_norm=batch_norm,
                                     kernel_init=kernel_init,
                                     pad=pad,
                                     m_pool=m_pool,
                                     bn_first=bn_first,
                                     ndims=ndims))
        filters *= 2
    # middle part
    input_tensor = encoder[-1][1]
    fully = ownkl.conv_layer(inputs=input_tensor, filters=filters, f_size=f_size,
                             activation=activation, batch_norm=batch_norm, kernel_init=kernel_init,
                             pad=pad, bn_first=bn_first, ndims=ndims)
    fully = Dropout(drop_3)(fully)
    fully = ownkl.conv_layer(inputs=fully, filters=filters, f_size=f_size,
                             activation=activation, batch_norm=batch_norm, kernel_init=kernel_init,
                             pad=pad, bn_first=bn_first, ndims=ndims)
    # build the decoder
    decoder.append(fully)
    for l in range(depth):
        # take the output of the previous decoder block and the output of the corresponding
        # encoder block
        input_lower = decoder[-1]
        input_skip = encoder.pop()[0]
        filters //= 2
        decoder.append(
            ownkl.upsampling_block(lower_input=input_lower,
                                   conv_input=input_skip,
                                   use_upsample=use_upsample,
                                   filters=filters,
                                   f_size=f_size,
                                   activation=activation,
                                   drop=dropouts.pop(),
                                   batch_norm=batch_norm,
                                   up_size=m_pool,
                                   bn_first=bn_first,
                                   ndims=ndims))

    # deep supervision
    # current test
    if supervision:
        try:
            UpSampling = getattr(ownkl, 'UpSampling{}DInterpol'.format(ndims))
        except Exception as e:
            logging.info('own Upsampling layer not available, fallback to tensorflow upsampling')
            UpSampling = getattr(kl, 'UpSampling{}D'.format(ndims))

        # mask from the pre-last upsampling block
        lower_mask = decoder[-2]
        lower_mask = Conv(filters_init, one_by_one, padding=pad, kernel_initializer=kernel_init, activation=activation)(
            lower_mask)

        lower_mask = UpSampling(size=m_pool)(lower_mask)

    outputs = decoder[-1]

    if supervision:
        # outputs = concatenate([lower_mask, outputs])
        outputs = Multiply()([lower_mask, outputs])

        # use either a 3x3 conv or 1x1
        # 16 or 4 filters
        # outputs = Conv(mask_classes, f_size, padding=pad, kernel_initializer=kernel_init, activation=activation)(outputs)

    return outputs


def get_model(config=dict(), metrics=None):
    """
    create a new model or load a pre-trained model
    :param config: json file
    :param metrics: list of tensorflow or keras metrics with gt,pred
    :return: returns a compiled keras model
    """

    # load a pre-trained model with config
    if config.get('LOAD', False):
        pass
        #return load_pretrained_model(config, metrics)

    # create a new 2D or 3D model with given config params
    return create_unet(config, metrics)


def test_unet():
    """
    Create a keras unet with a pre-configured config
    :return: prints model summary file
    """
    try:
        from src.utils.Utils_io import Console_and_file_logger
        Console_and_file_logger('test 2d network')
    except Exception as e:
        print("no logger defined, use print")

    config = {'GPU_IDS': '0', 'GPUS': ['/gpu:0'], 'EXPERIMENT': '2D/tf2/temp', 'ARCHITECTURE': '2D',
              'DIM': [224, 224], 'DEPTH': 4, 'SPACING': [1.0, 1.0], 'M_POOL': [2, 2], 'F_SIZE': [3, 3],
              'IMG_CHANNELS': 1, 'MASK_VALUES': [0, 1, 2, 3], 'MASK_CLASSES': 4, 'AUGMENT': False, 'SHUFFLE': True,
              'AUGMENT_GRID': True, 'RESAMPLE': False, 'DATASET': 'GCN_2nd', 'TRAIN_PATH': 'data/raw/GCN_2nd/2D/train/',
              'VAL_PATH': 'data/raw/GCN_2nd/2D/val/', 'TEST_PATH': 'data/raw/GCN_2nd/2D/val/',
              'DF_DATA_PATH': 'data/raw/GCN_2nd/2D/df_kfold.csv', 'MODEL_PATH': 'models/2D/tf2/gcn/2020-03-26_17_25',
              'TENSORBOARD_LOG_DIR': 'reports/tensorboard_logs/2D/tf2/gcn/2020-03-26_17_25',
              'CONFIG_PATH': 'reports/configs/2D/tf2/gcn/2020-03-26_17_25',
              'HISTORY_PATH': 'reports/history/2D/tf2/gcn/2020-03-26_17_25', 'GENERATOR_WORKER': 32, 'BATCHSIZE': 32,
              'INITIAL_EPOCH': 0, 'EPOCHS': 150, 'EPOCHS_BETWEEN_CHECKPOINTS': 5, 'MONITOR_FUNCTION': 'val_loss',
              'MONITOR_MODE': 'min', 'SAVE_MODEL_FUNCTION': 'val_loss', 'SAVE_MODEL_MODE': 'min', 'BN_FIRST': False,
              'OPTIMIZER': 'Adam', 'ACTIVATION': 'elu', 'LEARNING_RATE': 0.001, 'DECAY_FACTOR': 0.5, 'MIN_LR': 1e-10,
              'DROPOUT_L1_L2': 0.3, 'DROPOUT_L3_L4': 0.4, 'DROPOUT_L5': 0.5, 'BATCH_NORMALISATION': True,
              'USE_UPSAMPLE': True, 'LOSS_FUNCTION': keras.losses.binary_crossentropy}
    metrics = [tf.keras.losses.categorical_crossentropy]

    model = get_model(config, metrics)
    model.summary()


if __name__ == '__main__':
    test_unet()
