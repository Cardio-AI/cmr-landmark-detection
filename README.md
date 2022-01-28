Deep Learning based Landmark Detection Framework for CMR Images
==============================

This repository contains code to train and compare different **Deep Learning based landmark detection** models on 3D cardiac magnetic resonance (CMR) cine images.
The objective of these models is to detect the right ventricle insertion point (RVIP) in cine short axis (SAX) cardiac magnetic resonance (CMR) images. 
Furthermore, this repo was used in the paper **Comparison of Evaluation Metrics for Landmark Detection in CMR Images**[arxiv](https://arxiv.org/abs/2201.10410) to demonstrate very likely pitfalls of apparently simple detection and localisation metrics.



# Paper

Please cite the following paper if you use/modify or adapt part of the code from this repository:

    @misc{koehler2022comparison,
          title={Comparison of Evaluation Metrics for Landmark Detection in CMR Images}, 
          author={Sven Koehler and Lalith Sharan and Julian Kuhm and Arman Ghanaat and Jelizaveta Gordejeva and Nike K. Simon and Niko M. Grell and Florian André and Sandy Engelhardt},
          year={2022},
          eprint={2201.10410},
          archivePrefix={arXiv},
          primaryClass={cs.CV}
    }

[Arxiv-link](https://arxiv.org/abs/2201.10410)

**Abstract**

Cardiac Magnetic Resonance (CMR) images are widely used for cardiac diagnosis and ventricular assessment. Extracting specific landmarks like the right ventricular insertion points is of importance for spatial alignment and 3D modeling. The automatic detection of such landmarks has been tackled by multiple groups using Deep Learning, but relatively little attention has been paid to the failure cases of evaluation metrics in this field. In this work, we extended the public ACDC dataset with additional labels of the right ventricular insertion points and compare different variants of a heatmap-based landmark detection pipeline. In this comparison, we demonstrate very likely pitfalls of apparently simple detection and localisation metrics which highlights the importance of a clear detection strategy and the definition of an upper limit for localisation-based metrics. Our preliminary results indicate that a combination of different metrics is necessary, as they yield different winners for method comparison. Additionally, they highlight the need of a comprehensive metric description and evaluation standardisation, especially for the error cases where no metrics could be computed or where no lower/upper boundary of a metric exists


# Project Overview

- The repository dependencies are saved as conda environment (environment.yaml) file. 
- The Deep Learning models/layers are build with TF 2.X.
- Setup instruction for the repository are given here: [Install requirements](#Setup)
- An overview of all files and there usage is given here: [Repository Structure](#project-organization)


## Project-Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like 'make environment' or 'make requirement'
    ├── README.md          <- The top-level README for developers using this project.
    ├── data                     <- Ideally, dont save any data within the repo, if neccessary, use these folders
    │   ├── metadata       <- Excel and csv files with additional metadata
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── predicted      <- Model predictions, will be used for the evaluations
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── notebooks          <- Jupyter notebooks. 
    │   ├── Dataset        <- call the dataset helper functions, to analyze and slice the Dataset
    │   ├── Evaluate       <- See further below, reference to google-colab
    │   ├── Predict        <- Generate predictions for each fold
    │   ├── Train          <- Train a new model
    │
    ├── exp            <- Experiment folders, one exp-Folder per config file, one sub-folder per CV-split
    │                               Each exp-folder has the following files:
    │   ├── configs        <- Experiment config files as json
    │   ├── figures        <- Generated graphics and figures to be used in reporting
    │   ├── history        <- Tensorboard trainings history files
    │   ├── models         <- Trained and serialized models, model predictions, or model summaries
    │   ├── models.png     <- Model summary as picture
    │   └── tensorboard_logs  <- Generated figures/predictions, created while training to inspect the progress
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- Makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Make-data, train, and eval scripts & python modules with helpers
        ├── data           <- make_data script, Generator and pre-/post-processing utils
        ├── models         <- train-, predict- eval scripts, Model defnition and Tensorflow layers
        ├── utils          <- Metrics, losses, callbacks, io-utils, logger, notebook imports
        └── visualization  <- Plots for the data, generator or evaluations


## Datasets:
This repository uses the public available [ACDC dataset](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html). 

Within this work we extended the public dataset by binary labels of the right ventricular insertion points (RVIP). [link](https://heibox.uni-heidelberg.de/f/aa4baab97b78481a9bac/?dl=1)

Furthermore, we provide a dataframe with a pathology-based splitting of the ACDC patients which could be used for a 4-fold cross-validation. [link](https://heibox.uni-heidelberg.de/f/03f57e89dc8b46668144/?dl=1)

The ACDC dataset could be downloaded here: [acdc-download-link](https://acdc.creatis.insa-lyon.fr/#challenges).

Once you downloaded the original data, you can extend the official CMR images with our RVIP labels. 

We provide a Python [script](src/data/make_dataset.py)
and a [notebook](notebooks/Dataset/Prepare_data.ipynb) for data merging and 2D slice creation.

Both should do the same job and require:

- A local data-root path (we will write the merged data into this folder)
- A full-path to the unzipped original ACDC-dataset (path to the folder which lists all patient sub-folders)

According to the [STACOM Challenge 2012](http://stacom.cardiacatlas.org/lv-landmark-detection-challenge/landmark-points/), we defined the RVIP as the corner where the right ventricle joins the myocardium. 
Both IPs are segmented in the outermost corner of the right ventricle, such that one half is in the bright (ventricle), 
and the other half of the pixels is in the dark (myocardium) area. 
Slices close to the apex or base with uncertain placement positions were not labelled.

## Training

Our trainings script support single and multi-GPU training (data-parallelisms) and should run locally, and on clusters.
The trainings-flow is as follows:

1. Re-use or modify one of the example configs provided in exp/template_cfgs
2. Run src/models/train_model.py, which parse the following arguments:

            -cfg (Path (str) to an experiment config, you can find examples in exp/template_cfgs)

            -data (Path (str) to the data-root folder, please check src/Dataset/make_data.py or notebooks/Dataset/prepare_data.ipynb for further hints)

3. Our trainings script will sequentially train four models on the corresponding ACDC split. 
   The experiment config, model-definition/-weights, trainings-progress and tensorboard logs etc. will be saved automatically. 
   After each model convergence we call the predict_model on the corresponding fold and save the predicted files into each sub-folder.
   
4. Each experiment results in the following base-structure (note: repeating the same experiment config will create new time-step-based sub-folders):

      
The structure of our evaluation dataframe is described [here](#Evaluation).
```
├── df_eval.csv
├── _f0
│   └── 2021-10-21_20_38
├── _f1
│   └── 2021-10-21_21_33
├── _f2
│   └── 2021-10-21_22_29
└── _f3
    └── 2021-10-21_23_23
```
5. Each fold (_f0...,_f3) contains the following project files:

```
├── config (config used in this experiment fold)
├── gt (GT RVIP masks)
├── Log_errors.log (logging.error logfile)
├── Log.log (console and trainings progress logs)
├── model (model graph and weights for later usage)
├── model.png (graph as model png)
├── model_summary.txt (layer input/output shapes as structured txt file)
├── pred (predicted RVIP masks as nrrd files - same file names as GT)
└── tensorboard_logs (tensorboard logfiles: train-/test scalars and model predictions per epoch)
```

## Prediction
Usually the inference script will be called automatically per fold if we start the train_model.py file.
Nevertheless, you can also call our [predict](src/models/predict_model.py) script on other data or modify the experiment parameters.


This script takes two arguments:

            -exp (Path (str) to the root of one experiment fold)

            -data (Path (str) to the data-root folder, please check src/Dataset/make_data.py or notebooks/Dataset/prepare_data.ipynb for further hints)

Inference will load the experiment config, re-create the same model as used for training, load the final model weights and use the same pre-processing steps (Re-Sampling, cropping, scaling) as defined in the experiment config (config/config.json).

We post-process each prediction with the inverse of our pre-Processing operations and save the final prediction in the original image space and with the same image properties (spacing and dimension) as nrrd file.

## Evaluation
Each experiment results in a 4-fold cross-validation-sub-folder and one evaluation dataframe as csv. 
You can use the [evaluation notebook](notebooks/Evaluate/rvip_create_eval_plots.ipynb) for plotting and experiment comparison or process the results in Excel.

This dataframe has 

- one row per patient and labelled time-step (end diastole/end systole - ED/ES), and 
  
- 88 columns with some metadata and the slice-wise and volume based localisation and detection metrics

Subsequently, you can find some short hints about the column content.

**Column meanings**

File paths for the predictions, rvip-inter-observer-gt, orig-acdc-masks, and rvip-gt:

      'files_pred', 'files_io', 'files_orig_msk', 'files_gt',

patient id, cardiac phase, pathology, CMR spacings and experiment name:

      'patient', 'phase', 'pathology', 'spacing', 'inplane_spacing', 'EXP'

Right ventricular insertion points (RVIP) as lists per 3D vol:

      'ips_pred','ips_gt', 'ips_io', 'ips_orig_msk', 'ips_pred_single_also',

Mean RVIP and septums angle per 3D vol:

      'mips_pred', 'mips_gt', 'mips_io','mips_orig_msk', 'mips_pred_single_also', 'mangle_pred', 'mangle_gt', 'mangle_io', 'mangle_orig_msk',

Differences and mean difference between two angles (volume or slice-based):

      'diffs_gtpred','diffs_gtio', 'diffs_gtorig','mdiffs_gtpred', 'mdiffs_gtio', 'mdiffs_gtorig',

Distances (slice-wise) and mean distance (per volume) per ant/inf IP-pairs (pred,gt,io), allow/discard single point predictions, with upper limit/without:

      'dists_ant_gtpred', 'dists_inf_gtpred', 'dists_inf_gtpred','dists_ant_gtio','dists_inf_gtio', 'dists_ant_gtorig', 'dists_inf_gtorig', 'dists_ant_gtpred_single_also', 'dists_inf_gtpred_single_also', 'dists_ant_gtpred_up','dists_inf_gtpred_up','dists_ant_gtpred_single_also_up', 'dists_inf_gtpred_single_also_up', 'mdists_ant_gtpred', 'mdists_inf_gtpred', 'mdists_ant_gtio','mdists_inf_gtio', 'mdists_ant_gtorig', 'mdists_inf_gtorig', 'mdists_ant_gtpred_single_also','mdists_inf_gtpred_single_also', 'mdists_ant_gtpred_slice_wise', 'mdists_inf_gtpred_slice_wise', 'mdists_ant_gtpred_slice_wise_single_also','mdists_inf_gtpred_slice_wise_single_also', 'mdists_ant_gtpred_slice_wise_up', 'mdists_inf_gtpred_slice_wise_up', 'mdists_ant_gtpred_slice_wise_single_also_up','mdists_inf_gtpred_slice_wise_single_also_up'

Septums angle of each slice as list:

      'angles_pred', 'angles_gt', 'angles_io', 'angles_orig_msk',

True positive rates (TPR) and positive predictive value (PPV) per ant/inf IP:

      'tpr_ant', 'tpr_inf', 'tpr_ant_io','tpr_inf_io', 'tpr_ant_msk', 'tpr_inf_msk', 'ppv_ant', 'ppv_inf', 'ppv_ant_io', 'ppv_inf_io', 'ppv_ant_msk', 'ppv_inf_msk', 'tpr_ant_point', 'tpr_inf_point', 'ppv_ant_point', 'ppv_inf_point', 'tpr_ant_point_th15', 'tpr_inf_point_th15', 'ppv_ant_point_th15', 'ppv_inf_point_th15',


# Setup

Tested with Ubuntu 20.04

## Preconditions: 
- Python 3.6 locally installed 
(e.g.:  <a target="_blank" href="https://www.anaconda.com/download/#macos">Anaconda</a>)
- Installed nvidia drivers, cuda and cudnn 
(e.g.:  <a target="_blank" href="https://www.tensorflow.org/install/gpu">Tensorflow</a>)

## Local setup
Clone repository
```
git clone %repo-name%
cd %repo-name%
```

Create a conda environment from environment.yaml (environment name will be septum_detection)
```
conda env create --file environment.yaml
```

Activate environment
```
conda activate septum_detection
```
Install a helper to automatically change the working directory to the project root directory
```
pip install --extra-index-url https://test.pypi.org/simple/ ProjectRoot
```
Create a jupyter kernel from the activated environment, this kernel will be visible in the jupyter lab
```
python -m ipykernel install --user --name rvip --display-name "rvip kernel"


### Enable interactive widgets in Jupyterlab

Pre-condition: nodejs installed globally or into the conda environment. e.g.:
```
conda install -c conda-forge nodejs
```
Install the jupyterlab-manager which enables the use of interactive widgets
```
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

Further infos on how to enable the jupyterlab-extensions:
[JupyterLab](https://ipywidgets.readthedocs.io/en/latest/user_install.html#installing-the-jupyterlab-extension)
