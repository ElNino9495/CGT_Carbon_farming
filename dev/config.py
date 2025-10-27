import numpy as np
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'synthetic')
RESULTS_DIR = os.path.join(BASE_DIR, 'data', 'results', 'calibration')
DATA_FILENAME = "/Users/rohitsuresh/Documents/IISC/VCM_shapely/data/synthetic/farmer_data_log_normal.csv"

P = 1400.0          
COMMISSION = 0.3    
FIXED_COST = 30000.0 

PI_MIN = 20000.0    
PI_MAX = 50000.0    
LAMBDA_WEIGHT = 1.0 

MAX_COALITION_SIZE = 40
STRATEGIES = ['random', 'ascending', 'descending']
SEED = 42

GRID_BETA_RANGE = np.linspace(200, 1500, 15)
GRID_THETA_RANGE = np.linspace(0.3, 0.8, 15)

SCIPY_METHOD = 'L-BFGS-B'
SCIPY_BOUNDS = [(1.0, 5000.0), (0.01, 0.99)] # (beta, theta)

SHAPLEY_SAMPLES = 500 