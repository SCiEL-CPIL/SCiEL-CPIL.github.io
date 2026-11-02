"""Rewritten program for Matrix multiplication 2,2,11 rank minimization."""
import numpy as np
from scipy.optimize import linprog
import itertools
from collections import defaultdict

DIMENSIONS = [2, 2, 11]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x11 matrix.
    
    Returns the schoolbook decomposition with rank 44.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
