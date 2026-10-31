import numpy as np
import time

N = 600
DELTA = 1.0 / 1200  # interval width

def compute_score(h):
    # h: array of 600 heights
    # Compute max of autoconvolution
    # Use fine grid
    n_grid = 4096
    # Create step function on fine grid
    # The step function is on [-1/4, 1/4]
    # Fine grid points
    x = np.linspace(-0.25, 0.25, n_grid, endpoint=False) + DELTA/2  # centers
    # Actually, let's use the intervals
    # Interval i covers [x0 + i*DELTA, x0 + (i+1)*DELTA] where x0 = -0.25
    # For the fine grid, assign each grid point to the nearest interval
    # Simpler: create the step function directly
    f = np.repeat(h, n_grid // N)  # approximate
    # Better: use interpolation
    # Actually, let's compute the autoconvolution analytically for step functions
    pass
