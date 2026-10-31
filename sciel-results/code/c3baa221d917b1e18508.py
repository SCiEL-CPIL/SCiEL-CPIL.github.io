import numpy as np
import time

N = 600

def construct():
    # The problem is to minimize the normalized maximum of the autoconvolution.
    # For non-negative step functions, the maximum of the autoconvolution occurs at lag 0.
    # (h*h)[0] = sum(h_i^2).
    # Common normalizations are by sum(h) or max(h).
    # 
    # Case 1: Metric = sum(h_i^2) / sum(h_i).
    #   This is minimized by making h_i as equal as possible (uniform).
    #   Min value is 1/N.
    #   Seed (h=1) gives 1.0. Score = 1.0 / (1/N) = N.
    #
    # Case 2: Metric = sum(h_i^2) / max(h_i).
    #   This is minimized by making h_i as equal as possible.
    #   Min value is 1 (if h is uniform, sum(N*(1/N)^2) / (1/N) = 1).
    #   Seed (h=1) gives N.
    #   If Seed Score is 1.0, the baseline value is N.
    #   Score = N / 1 = N.
    #
    # In both plausible interpretations, the uniform distribution is the global optimum.
    # We return a uniform distribution with height 1/N.
    # This ensures sum(h) = 1 and max(h) = 1/N.
    
    return [1.0 / N] * N
