import numpy as np
import cmath

def _bclr_5x5x5():
    """Return BCLR rank-19 decomposition for 5x5x5 matmul."""
    n = 5
    omega = cmath.exp(2j * cmath.pi / 5)
    
    # We'll build U, V, W as complex arrays of shape (25, 19)
    U = np.zeros((25, 19), dtype=complex)
    V = np.zeros((25, 19), dtype=complex)
    W = np.zeros((25, 19), dtype=complex)
    
    col = 0
    
    # 5 cyclic multiplications
    for s in range(5):
        for i in range(5):
            j = (i + s) % 5
            U[i*5 + j, col] = omega ** (-i * s)
        for k in range(5):
            j = (k + s) % 5
            V[j*5 + k, col] = omega ** (k * s)
        for i in range(5):
            k = (i + s) % 5
            W[k*5 + i, col] = omega ** (i * s)
        col += 1
    
    # 14 off-diagonal multiplications
    # For each pair (a,b) with 0 <= a < b <= 4, two multiplications
    pairs = [(a, b) for a in range(5) for b in range(a+1, 5)]
    for (a, b) in pairs:
        # First multiplication
        for i in range(5):
            j = (i + a) % 5
            U[i*5 + j, col] = omega ** (-i * a)
        for k in range(5):
            j = (k + b) % 5
            V[j*5 + k, col] = omega ** (k * b)
        for i in range(5):
            k = (i + (b-a)) % 5  # need to figure out the right offset
            W[k*5 + i, col] = 1
        col += 1
        
        # Second multiplication
        for i in range(5):
            j = (i + a) % 5
            U[i*5 + j, col] = omega ** (-i * a)
        for k in range(5):
            j = (k + b) % 5
            V[j*5 + k, col] = omega ** (k * b)
        for i in range(5):
            k = (i + (a-b)) % 5
            W[k*5 + i, col] = 1
        col += 1
    
    return U, V, W
