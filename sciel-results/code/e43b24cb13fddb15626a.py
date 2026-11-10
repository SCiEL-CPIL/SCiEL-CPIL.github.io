import numpy as np

def kakeya_needle_3d8() -> np.ndarray:
       N = 8
       positions = np.zeros((64, 2))
       for k in range(64):
           i = k // 8
           j = k % 8
           positions[k, 0] = -i / 8.0
           positions[k, 1] = -j / 8.0
       return positions
