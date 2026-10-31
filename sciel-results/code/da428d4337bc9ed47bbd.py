import numpy as np

N = 600

def construct():
    # Interval width
    w = 0.5 / N
    # Number of spikes (every other interval)
    k = N // 2
    # Height of each spike to ensure total L1 norm is 1
    # k * h * w = 1 => h = 1 / (k * w)
    h = 1.0 / (k * w)
    
    heights = [0.0] * N
    for i in range(0, N, 2):
        heights[i] = h
        
    return heights
