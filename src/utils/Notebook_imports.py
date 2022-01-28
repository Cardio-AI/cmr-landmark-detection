# central way to get all standard library available in one import
SEED = 42
import logging
import numpy as np
import pandas as pd
from logging import info
import os
import json
import glob
import datetime
from ipywidgets import interact
from ipywidgets import interact_manual
import SimpleITK as sitk
import random
from collections import Counter
import seaborn as sb
import matplotlib.pyplot as plt
import random
random.seed(SEED)
np.random.seed(SEED)

from src.utils.Utils_io import Console_and_file_logger, ensure_dir

# make jupyter able to display multiple lines of variables in one cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "last"

mpl_logger = logging.getLogger('matplotlib') 
mpl_logger.setLevel(logging.WARNING) 
