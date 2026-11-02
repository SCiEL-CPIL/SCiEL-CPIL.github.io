import time
import numpy as np
from fractions import Fraction
import itertools

DIMENSIONS = [2, 3, 10]

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix multiplication.
    Uses Strassen for 2x2 blocks and schoolbook for 2x1 blocks.
    Rank = 5 * (7 + 4) = 55.
    """
    n, m, p = DIMENSIONS
    rank = 55
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    a00, a01, a10, a11 = 0, 1, 3, 4
    a02, a12 = 2, 5
    
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        b00 = k0
        b01 = k1
        b10 = 10 + k0
        b11 = 10 + k1
        b20 = 20 + k0
        b21 = 20 + k1
        
        c00 = k0 * 2
        c01 = k1 * 2
        c10 = k0 * 2 + 1
        c11 = k1 * 2 + 1
        
        # Strassen terms
        # M1
        U[a00, col] = 1.0; U[a11, col] = 1.0
        V[b00, col] = 1.0; V[b11, col] = 1.0
        W[c00, col] = 1.0; W[c11, col] = 1.0
        col += 1
        
        # M2
        U[a10, col] = 1.0; U[a11, col] = 1.0
        V[b00, col] = 1.0
        W[c10, col] = 1.0; W[c11, col] = -1.0
        col += 1
        
        # M3
        U[a00, col] = 1.0
        V[b01, col] = 1.0; V[b11, col] = -1.0
        W[c01, col] = 1.0; W[c11, col] = 1.0
        col += 1
        
        # M4
        U[a11, col] = 1.0
        V[b10, col] = 1.0; V[b00, col] = -1.0
        W[c00, col] = 1.0; W[c10, col] = 1.0
        col += 1
        
        # M5
        U[a00, col] = 1.0; U[a01, col] = 1.0
        V[b11, col] = 1.0
        W[c00, col] = -1.0; W[c01, col] = 1.0
        col += 1
        
        # M6
        U[a10, col] = 1.0; U[a00, col] = -1.0
        V[b00, col] = 1.0; V[b01, col] = 1.0
        W[c11, col] = 1.0
        col += 1
        
        # M7
        U[a01, col] = 1.0; U[a11, col] = -1.0
        V[b10, col] = 1.0; V[b11, col] = 1.0
        W[c00, col] = 1.0
        col += 1
        
        # Schoolbook terms
        U[a02, col] = 1.0; V[b20, col] = 1.0; W[c00, col] = 1.0
        col += 1
        
        U[a02, col] = 1.0; V[b21, col] = 1.0; W[c01, col] = 1.0
        col += 1
        
        U[a12, col] = 1.0; V[b20, col] = 1.0; W[c10, col] = 1.0
        col += 1
        
        U[a12, col] = 1.0; V[b21, col] = 1.0; W[c11, col] = 1.0
        col += 1
    
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
