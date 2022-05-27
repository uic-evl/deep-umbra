import cython
import warnings

import numpy as np
import pyximport

warnings.filterwarnings('ignore', '.*deprecated NumPy.*')
pyximport.install(
    setup_args={'include_dirs': np.get_include(), },
    reload_support=True,
)
from .functions import (
    load_image,
    num2deg,
    deg2num,
    degs2nums,
    nums2degs,
)
# from shadow.cutil.functions import (
#     load_image,
#     num2deg,
#     deg2num,
#     degs2nums,
#     nums2degs,
# )
