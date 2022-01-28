# -*- coding: utf-8 -*-
import glob
import os

from src.data.Dataset import ensure_dir, create_2d_slices_from_3d_volume_files


# Download and unpack the raw data
# small helper
def clean_import(dir_path):
    import shutil
    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        print("Error: %s : %s" % (dir_path, e.strerror))
        print('Dont worry, irectory will be created.')
    ensure_dir(dir_path)


def main(data_root, path_to_acdc_original):
    # ------------------------------------------define logging and working directory
    from ProjectRoot import change_wd_to_project_root
    change_wd_to_project_root()

    # define a folder for the acdc cmr and masks, make sure not to use an existing folder
    import_path = os.path.join(data_root, 'import')
    ensure_dir(data_root)
    clean_import(import_path)

    # download cleaned rvip 3D cmr and masks
    url = 'https://heibox.uni-heidelberg.de/f/8776d7311ec84723aacf/?dl=1 '
    os.system('wget {} -P {}'.format(url, import_path))
    print('downloaded')
    # unzip and replace
    zip_file = glob.glob(os.path.join(import_path, 'index.html?dl=*'))[0]
    os.system('unzip -o {} -d {}'.format(zip_file, data_root))
    # clean temp import older
    clean_import(import_path)

    # remove old and download new cv-dataframe
    os.system('rm {} -f'.format(os.path.join(data_root, 'df_kfold.csv')))
    url2 = 'https://heibox.uni-heidelberg.de/f/03f57e89dc8b46668144/?dl=1'
    os.system('wget {} -P {}'.format(url2, import_path))
    print('downloaded')
    # unzip and replace
    zip_file = glob.glob(os.path.join(import_path, 'index.html?dl=*'))[0]
    os.system('unzip -o {} -d {}'.format(zip_file, data_root))
    # clean temp import folder
    clean_import(import_path)

    # io == interobserver
    # pp == 100 patients x phases xrvip/cmr = 400 files
    print('collect 3D CMR from: {}'.format(path_to_acdc_original))
    # searches in all patient folders for any 3D CMR (2 frames per patient) as nifti
    images = sorted(glob.glob(os.path.join(path_to_acdc_original, '*/*frame[0-9][0-9].nii.gz')))
    print('images: {}'.format(len(images)))

    # quality check of the image and mask names, find names with wrong names
    # give input and output path here
    input_path = os.path.join(data_root, 'pp')
    export_path = os.path.join(data_root, '2D')
    # images = sorted(glob.glob(os.path.join(input_path, '*frame[0-9][0-9].nii')))
    masks = sorted(glob.glob(os.path.join(input_path,
                                          '*frame[0-9][0-9]_rvip.nrrd')))  # searches in all first level folders for any mask as nrrd
    print('images: {}'.format(len(images)))
    print('masks: {}'.format(len(masks)))
    assert (len(images) == len(masks)), 'len(images)-> {} != len(masks)-> {} '.format(len(images), len(masks))
    # in the optimal case there should be as many images as masks. If not, some of the annotations might have been saved with a wrong name.

    # Slice the 3D vol in 2D slices
    ensure_dir(export_path)
    _ = [create_2d_slices_from_3d_volume_files(img_f=img, mask_f=msk, export_path=export_path) for img, msk in
         zip(images, masks)]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='train a RV IP detection/segmentation model on CMR images')

    # usually these two parameters should encapsulate all experiment parameters
    parser.add_argument('-data_root', action='store', default='data/import')
    parser.add_argument('-acdc_data', action='store', default='data/import/original')

    results = parser.parse_args()
    print('given parameters: {}'.format(results))
    data_root = results.data_root
    acdc_data = results.acdc_data
    try:
        main(data_root, acdc_data)
    except Exception as e:
        print(e)
    exit()
