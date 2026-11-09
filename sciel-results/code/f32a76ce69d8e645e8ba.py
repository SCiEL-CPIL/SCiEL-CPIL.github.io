import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _validate(U, V, W, n, m, p):
    dim_a = n * m
    dim_b = m * p
    dim_c = p * n
    rank = len(U[0])
    for a in range(dim_a):
        for b in range(dim_b):
            for c in range(dim_c):
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                expected = 1 if (i == i2 and j == j2) else 0
                actual = Fraction(0)
                for col in range(rank):
                    actual += Fraction(U[a][col]) * Fraction(V[b][col]) * Fraction(W[c][col])
                if actual != expected:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    
    # Return schoolbook decomposition
    return _schoolbook(n, m, p)
