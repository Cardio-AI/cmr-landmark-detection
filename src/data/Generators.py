import concurrent.futures
import logging
import os
import platform
import random
from concurrent.futures import as_completed
from random import choice
from time import time

import SimpleITK as sitk
# from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras

from src.data.Dataset import describe_sitk, copy_meta_and_save, create_2d_slices_from_4d_volume_file
from src.data.Preprocess import match_2d_on_nd as mhist
from src.data.Preprocess import resample_3D, clip_quantile, normalise_image, \
    transform_to_binary_mask, load_masked_img, augmentation_compose_2d_3d_4d, pad_and_crop, calc_resampled_size
from src.visualization.Visualize import show_2D_or_3D


#    get_patient, get_img_msk_files_from_split_dir


class BaseGenerator(tensorflow.keras.utils.Sequence):
    """
    Base generator class
    """

    def __init__(self, x=None, y=None, config=None, in_memory=False):
        """
        Creates a datagenerator for a list of nrrd images and a list of nrrd masks
        :param x: list of nrrd image file names
        :param y: list of nrrd mask file names
        :param config:
        """

        if config is None:
            config = {}
        if y is None:
            self.MASKS = False
            self.SINGLE_OUTPUT = True
        logging.info('Create DataGenerator')

        if y is not None:  # return x, y
            assert (len(x) == len(y)), 'len(X) != len(Y)'

        def normalise_paths(elem):
            """
            recursive helper to clean filepaths, could handle list of lists and list of tuples
            """
            if type(elem) in [list, tuple]:
                return [normalise_paths(el) for el in elem]
            elif isinstance(elem, str):
                return os.path.normpath(elem)
            else:
                return elem

        # linux/windows cleaning
        if platform.system() == 'Linux':
            x = normalise_paths(x)
            if self.MASKS: y = normalise_paths(y)

        self.INDICES = list(range(len(x)))
        # override if necessary
        #self.SINGLE_OUTPUT = config.get('SINGLE_OUTPUT', False)

        self.IMAGES = x
        self.LABELS = y

        # if streamhandler loglevel is set to debug, print each pre-processing step
        self.DEBUG_MODE = logging.getLogger().handlers[1].level == logging.DEBUG
        # self.DEBUG_MODE = False

        # read the config, set default values if param not given
        self.SCALER = config.get('SCALER', 'MinMax')
        self.AUGMENT = config.get('AUGMENT', False)
        self.AUGMENT_PROB = config.get('AUGMENT_PROB', 0.8)
        self.SHUFFLE = config.get('SHUFFLE', True)
        self.RESAMPLE = config.get('RESAMPLE', False)
        self.SPACING = config.get('SPACING', [1.25, 1.25])
        self.SEED = config.get('SEED', 42)
        self.DIM = config.get('DIM', [256, 256])
        self.BATCHSIZE = config.get('BATCHSIZE', 32)
        self.MASK_VALUES = config.get('MASK_VALUES', [0, 1, 2, 3])
        self.N_CLASSES = len(self.MASK_VALUES)
        # create one worker per image & mask (batchsize) for parallel pre-processing if nothing else is defined
        self.MAX_WORKERS = config.get('GENERATOR_WORKER', self.BATCHSIZE)
        self.MAX_WORKERS = min(32, self.MAX_WORKERS)
        self.IN_MEMORY = in_memory
        if self.DEBUG_MODE:
            self.MAX_WORKERS = 1  # avoid parallelism when debugging, otherwise the plots are shuffled
        self.THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=self.MAX_WORKERS)

        if not hasattr(self, 'X_SHAPE'):
            self.X_SHAPE = np.empty((self.BATCHSIZE, *self.DIM, 1), dtype=np.float32)
            self.Y_SHAPE = np.empty((self.BATCHSIZE, *self.DIM, self.N_CLASSES), dtype=np.float32)

        logging.info(
            'Datagenerator created with: \n shape: {}\n spacing: {}\n batchsize: {}\n Scaler: {}\n Images: {} \n Augment: {} \n Thread workers: {}'.format(
                self.DIM,
                self.SPACING,
                self.BATCHSIZE,
                self.SCALER,
                len(
                    self.IMAGES),
                self.AUGMENT,
                self.MAX_WORKERS))

        self.on_epoch_end()

    def __plot_state_if_debug__(self, img, mask=None, start_time=None, step='raw'):

        if self.DEBUG_MODE:

            try:
                logging.debug('{}:'.format(step))
                logging.debug('{:0.3f} s'.format(time() - start_time))
                describe_sitk(img)
                describe_sitk(mask)
                if self.MASKS:
                    show_2D_or_3D(img, mask)
                    plt.show()
                else:
                    show_2D_or_3D(img)
                    plt.show()
                    # maybe this crashes sometimes, but will be caught
                    if mask:
                        show_2D_or_3D(mask)
                        plt.show()

            except Exception as e:
                logging.debug('plot image state failed: {}'.format(str(e)))

    def __len__(self):

        """
        Denotes the number of batches per epoch
        :return: number of batches
        """
        return int(np.floor(len(self.INDICES) / self.BATCHSIZE))

    def __getitem__(self, index):

        """
        Generate indexes for one batch of data
        :param index: int in the range of  {0: len(dataset)/Batchsize}
        :return: pre-processed batch
        """

        t0 = time()
        # collect n x indexes with n = Batchsize
        # starting from the given index parameter
        # which is in the range of  {0: len(dataset)/Batchsize}
        idxs = self.INDICES[index * self.BATCHSIZE: (index + 1) * self.BATCHSIZE]

        # Collects the value (a list of file names) for each index
        # list_IDs_temp = [self.LIST_IDS[k] for k in idxs]
        logging.debug('index generation: {}'.format(time() - t0))
        # Generate data
        return self.__data_generation__(idxs)

    def on_epoch_end(self):

        """
        Recreates and shuffle the indexes after each epoch
        :return: None
        """

        self.INDICES = np.arange(len(self.INDICES))
        if self.SHUFFLE:
            np.random.shuffle(self.INDICES)

    def __data_generation__(self, idxs):

        """
        Generates data containing batch_size samples

        :param list_IDs_temp:
        :return: X : (batchsize, *dim, n_channels), Y : (batchsize, *dim, number_of_classes)
        """

        # Initialization

        x = np.empty_like(self.X_SHAPE)
        y = np.empty_like(self.Y_SHAPE)

        futures = set()

        t0 = time()
        # Generate data
        for i, ID in enumerate(idxs):
            try:
                # remember the ordering of the shuffled indexes,
                # otherwise files, that take longer are always at the batch end
                futures.add(self.THREAD_POOL.submit(self.__preprocess_one_image__, i, ID))

            except Exception as e:
                PrintException()
                print(e)
                logging.error(
                    'Exception {} in datagenerator with: image: {} or mask: {}'.format(str(e), self.IMAGES[ID],
                                                                                       self.LABELS[ID]))
                raise e  # testing phase --> make sure all errors are handled

        for i, future in enumerate(as_completed(futures)):
            # use the indexes i to place each processed example in the batch
            # otherwise slower images will always be at the end of the batch
            # Use the ID for exception handling as reference to the file name
            try:
                x_, y_, i, ID, needed_time = future.result()
                if self.SINGLE_OUTPUT:
                    x[i,], _ = x_, y_
                else:
                    x[i,], y[i,] = x_, y_
                logging.debug('img finished after {:0.3f} sec.'.format(needed_time))
            except Exception as e:
                logging.error(
                    'Exception {} in datagenerator with: image: {} or mask: {}'.format(str(e), self.IMAGES[ID],
                                                                                       self.LABELS[ID]))

        logging.debug('Batchsize: {} preprocessing took: {:0.3f} sec'.format(self.BATCHSIZE, time() - t0))
        if self.SINGLE_OUTPUT:
            #x,None
            return x.astype(np.float32), y
        else:
            return np.array(x.astype(np.float32)), np.array(y.astype(np.float32))

    def __preprocess_one_image__(self, i, ID):
        logging.error('not implemented error')


class DataGenerator(BaseGenerator):
    """
    Yields (X, Y) / image,mask for 2D and 3D U-net training
    could be used to yield (X, None)
    """

    def __init__(self, x=None, y=None, config=None, in_memory=False):
        if config is None:
            config = {}
        self.MASKING_IMAGE = config.get('MASKING_IMAGE', False)
        self.SINGLE_OUTPUT = False
        self.MASKING_VALUES = config.get('MASKING_VALUES', [1, 2, 3])
        self.HIST_MATCHING = config.get('HIST_MATCHING', False)
        self.IMG_INTERPOLATION = config.get('IMG_INTERPOLATION', sitk.sitkLinear)
        self.MSK_INTERPOLATION = config.get('MSK_INTERPOLATION', sitk.sitkNearestNeighbor)
        self.GAUS = config.get('GAUS', False)  # apply gaus smoothing
        self.SIGMA = config.get('SIGMA', 1)  # gaus sigma value
        self.IN_MEMORY = in_memory
        self.config = config

        # how to get from image path to mask path
        # the wildcard is used to load a mask and cut the images by one or more labels
        self.REPLACE_DICT = {}
        GCN_REPLACE_WILDCARD = ('img', 'msk')
        ACDC_REPLACE_WILDCARD = ('.nii.gz', '_gt.nii.gz')

        if 'ACDC' in x[0]:
            self.REPLACE_WILDCARD = ACDC_REPLACE_WILDCARD
        else:
            self.REPLACE_WILDCARD = GCN_REPLACE_WILDCARD
        # if masks are given
        if y is None:
            self.MASKS = False
            logging.info('inference mode, no masks given, will use x as placeholder for y in fix processing' )
        else:
            self.MASKS = True

        super().__init__(x=x, y=y, config=config, in_memory=in_memory)

        # in memory preprocessing for the cluster
        # in memory training for the cluster
        if self.IN_MEMORY:
            print('in memory preprocessing')
            zipped = list()
            futures = [self.THREAD_POOL.submit(self.__fix_preprocessing__, i) for i in range(len(self.IMAGES))]
            for i, future in enumerate(as_completed(futures)):
                zipped.append(future.result())
            self.IMAGES_PROCESSED, self.LABELS_PROCESSED = list(map(list, zip(*zipped)))

    def __fix_preprocessing__(self, ID):

        t0 = time()

        # load image
        sitk_img = load_masked_img(sitk_img_f=self.IMAGES[ID], mask=self.MASKING_IMAGE,
                                   masking_values=self.MASKING_VALUES, replace=self.REPLACE_WILDCARD)
        if self.MASKS:
            # load mask
            sitk_msk = load_masked_img(sitk_img_f=self.LABELS[ID], mask=self.MASKING_IMAGE,
                                       masking_values=self.MASKING_VALUES, replace=self.REPLACE_WILDCARD,
                                       mask_labels=self.MASK_VALUES)
        else:
            sitk_msk = sitk_img

        self.__plot_state_if_debug__(sitk_img, sitk_msk, t0, 'raw')
        t1 = time()

        if self.RESAMPLE:
            if sitk_img.GetDimension() in [2, 3]:
                # transform the spacing from numpy representation towards the sitk representation
                target_spacing = list(reversed(self.SPACING))
                new_size_inputs = calc_resampled_size(sitk_img, target_spacing)

                logging.debug('dimension: {}'.format(sitk_img.GetDimension()))
                logging.debug('Size before resample: {}'.format(sitk_img.GetSize()))

                sitk_img = resample_3D(sitk_img=sitk_img,
                                       size=new_size_inputs,
                                       spacing=target_spacing,
                                       interpolate=self.IMG_INTERPOLATION)
                # CHANGED
                sitk_msk = resample_3D(sitk_img=sitk_msk,
                                       size=new_size_inputs,
                                       spacing=target_spacing,
                                       interpolate=self.MSK_INTERPOLATION)

                logging.debug('Spacing after resample: {}'.format(sitk_img.GetSpacing()))
                logging.debug('Size after resample: {}'.format(sitk_img.GetSize()))
                logging.debug('spatial resampling took: {:0.3f} s'.format(time() - t1))
                t1 = time()

            else:
                raise NotImplementedError('dimension not supported: {}'.format(sitk_img.GetDimension()))
        # transform to nda for further processing
        img_nda = sitk.GetArrayFromImage(sitk_img)
        mask_nda = sitk.GetArrayFromImage(sitk_msk)

        self.__plot_state_if_debug__(img_nda, mask_nda, t1, 'resampled')
        t1 = time()

        # We need to normalise the image/before augmentation, albumentation expects them to be normalised
        img_nda = clip_quantile(img_nda, .999)
        img_nda = normalise_image(img_nda, normaliser=self.SCALER)

        if not self.MASKS:  # yields the image two times for an autoencoder
            mask_nda = clip_quantile(mask_nda, .999)
            mask_nda = normalise_image(mask_nda, normaliser=self.SCALER)
            # mask_nda = normalise_image(mask_nda, normaliser=self.SCALER)

        self.__plot_state_if_debug__(img_nda, mask_nda, t1, '{} normalized image:'.format(self.SCALER))
        return img_nda, mask_nda

    def __preprocess_one_image__(self, i, ID):
        t0 = time()
        border = 2
        ref = None
        apply_hist_matching = self.HIST_MATCHING and random.random() < 0.1 # apply histmatching only in les than 10% of the cases
        if apply_hist_matching:
            if hasattr(self, 'IMAGES_PROCESSED'):
                ref = choice(self.IMAGES_PROCESSED)
            else:
                ref = sitk.GetArrayFromImage(sitk.ReadImage(choice(self.IMAGES)))
            #ref = pad_and_crop(ref, target_shape=self.DIM) # this is not resampled, safe computation time
            if ref.ndim == 3:  # for 2D we dont need to select one slice
                ref = ref[choice(list(range(ref.shape[0] - 1))[border:-border])]  # choose o random slice as reference

        if self.IN_MEMORY:
            img_nda, mask_nda = self.IMAGES_PROCESSED[ID], self.LABELS_PROCESSED[ID]
        else:
            img_nda, mask_nda = self.__fix_preprocessing__(ID)
        t1 = time()

        if self.AUGMENT:  # augment data with albumentation
            if apply_hist_matching:
                img_nda = mhist(img_nda, ref)

            # use albumentation to apply random rotation scaling and shifts
            img_nda, mask_nda = augmentation_compose_2d_3d_4d(img_nda, mask_nda, probabillity=self.AUGMENT_PROB, config=self.config)

            self.__plot_state_if_debug__(img_nda, mask_nda, t1, 'augmented')
            t1 = time()

        img_nda, mask_nda = map(lambda x: pad_and_crop(x, target_shape=self.DIM),
                                [img_nda, mask_nda])

        img_nda = normalise_image(img_nda, normaliser=self.SCALER)

        # transform the labels to binary channel masks
        # if masks are given, otherwise keep image as it is (for vae models, masks == False)
        if self.MASKS:
            mask_nda = transform_to_binary_mask(mask_nda, self.MASK_VALUES)
            if self.GAUS:
                import scipy.ndimage
                mask_nda = np.stack(
                    [scipy.ndimage.gaussian_filter(mask_nda[..., c].astype(np.float32), self.SIGMA)
                     for c in range(mask_nda.shape[-1])],
                    axis=-1)
                mask_nda = normalise_image(mask_nda, normaliser='minmax')
        else:  # yields two images
            mask_nda = normalise_image(mask_nda, normaliser=self.SCALER)
            mask_nda = mask_nda[..., np.newaxis]

        self.__plot_state_if_debug__(img_nda, mask_nda, t1, 'after crop')

        return img_nda[..., np.newaxis], mask_nda, i, ID, time() - t0


def sliceable(generator, temp_path='data/interim', **args):
    """
        Annotation wrapper, that creates t * z 2D generators from a 4D CMR file.
    This enable to inference a complete 4D CMR sequence on a 2D segmentation model

    Usage:
    gens = sliceable(DataGenerator,x=files_filtered,y=None, config=pred_config)
    """
    temp_path = 'data/interim'

    x = args.get('x', None)
    y = args.get('y', None)
    cfg = args.get('config', {})
    cfg['BATCHSIZE'] = 1
    dim = sitk.ReadImage(x[0]).GetDimension()
    if dim == 4:
        logging.info('found {} 4D files, will return one generator per file with t x z slices'.format(len(x)))
        generators = []
        for i in range(len(x)):
            x_sliced = create_2d_slices_from_4d_volume_file(x[i], temp_path)
            if y is not None: y_sliced = create_2d_slices_from_4d_volume_file(y[i], temp_path)
            logging.info('x_sliced: {}, example: {}'.format(len(x_sliced), x_sliced[0]))
            generators.append(generator(x=x_sliced, y=None, config=cfg))
    return generators

import linecache
import sys


def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))
