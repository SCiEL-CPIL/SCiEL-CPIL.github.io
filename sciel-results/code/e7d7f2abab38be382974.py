import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14
START_TIME = time.time()
TIME_LIMIT = 110.0 # Safety margin

def get_score(points):
    # ... calculation ...
    pass

def smooth_objective(points, tau=100.0):
    # ... log-sum-exp calculation ...
    pass

def construct():
    # ... multi-start optimization ...
    pass
