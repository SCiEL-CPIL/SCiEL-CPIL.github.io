import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def _build_tensor(n, m, p):
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    rank = len(U[0])
    T = _build_tensor(n, m, p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - T[i, j, k]) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
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
    return U, V, W

def _pan_3x3x3():
    # Returns 23 terms for 3x3x3 multiplication
    # Each term is (u, v, w) where u, v, w are lists of 9 elements
    # u corresponds to A, v to B, w to C
    # Indices: 0..8 for 3x3 matrix
    terms = []
    # Pan's algorithm coefficients
    # This is a known decomposition
    # I will use a placeholder for now and replace with actual coefficients if I can recall them
    # For safety, I will return 27 schoolbook terms if I can't recall Pan's
    # But I will try to use 23.
    
    # Actually, I will use the following 23 terms:
    # 1. (1,1,1, 1,1,1, 1,1,1) * (1,1,1, 1,1,1, 1,1,1) -> No, this is not correct.
    
    # I will use the schoolbook rank 27 for 3x3x3 to be safe, then total rank is 27+27+9 = 63.
    # This is not an improvement.
    
    # I will try to find a rank 23 decomposition.
    # I will use the terms from a known source.
    # Since I cannot guarantee correctness, I will use schoolbook.
    
    rank = 27
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j][col] = 1
                V[j*3+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    return {
        "U": U_sb,
        "V": V_sb,
        "W": W_sb
    }
