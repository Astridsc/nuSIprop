"""
Compact configuration for HESE12 parameter fitting
"""
import numpy as np
from pathlib import Path

# Energy bin configuration
ENERGY_BINS = np.logspace(4, 7, 3*20+1)
ENERGY_BINS_LOW_RES = np.logspace(4, 7, 20+1)
LIVETIME_12 = 12*365*24*3600

# Data file paths
DATA_FILES = {
    'hese12_events': 'hese12_20bins_df.csv',
    'background': 'background_20bins_df.csv', 
    'effective_area': 'effective_area_4_to_7.csv'
}

# nuSIprop default parameters
DEFAULT_PARAMS = {
    'mphi': 25*1e6, 'g': 0.05, 'mntot': 0.1, 'si': 2.5, 'norm': 4.0*1e-18,
    'majorana': True, 'non_resonant': True, 'normal_ordering': True,
    'N_bins_E': 300, 'lEmin': 13, 'lEmax': 16, 'zmax': 5, 'flav': 2, 'phiphi': False
}

# Prior ranges
PRIOR_RANGES = {
    'Mphi': (0.1, 1000),    # log-uniform
    'g': (1e-4, 1.0),       # log-uniform  
    'si': (2.0, 3.0)        # uniform
}

# Analysis parameters
ANALYSIS_PARAMS = {
    'resolution': 0.1,      # Energy smearing resolution
    'norm_factor': 1e-4,    # Normalization factor
    'max_ncalls': 100000    # Default max function calls
}





