import numpy as np
import time

N = 400

def construct():
    # Deadline guard
    start = time.time()
    deadline = 110.0 # 10s margin
    
    # Generate a smooth window to minimize off-peak correlations
    # If the metric includes peak, this might be suboptimal vs ones,
    # but ones failed. Smoothing is a common fix for "correlation" tasks.
    i = np.arange(N)
    center = (N - 1) / 2.0
    sigma = N / 4.0
    f = np.exp(-((i - center) ** 2) / (2 * sigma ** 2))
    
    # Ensure it's a list of floats
    return [float(x) for x in f]
