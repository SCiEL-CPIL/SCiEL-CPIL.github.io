import numpy as np

N = 600

def construct():
    heights = np.zeros(N)
    heights[::2] = 1.0
    return heights.tolist()
