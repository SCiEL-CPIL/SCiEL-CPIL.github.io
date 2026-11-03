import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 3]
DEADLINE = 150.0  # wall-clock limit
START = time.time()

def elapsed():
    return time.time() - START

def time_left():
    return DEADLINE - elapsed()

def construct():
    # ... build rank-17 baseline, then try to improve
    ...
