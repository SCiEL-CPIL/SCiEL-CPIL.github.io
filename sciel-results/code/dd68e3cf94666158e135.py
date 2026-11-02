import numpy as np
import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
                
    # Try to find a better decomposition for 2x3x3 blocks
    # If found, combine two of them
    # For now, return schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
