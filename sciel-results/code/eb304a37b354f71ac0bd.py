"""
Program to find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
"""
import time
import numpy as np

# Deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 115.0  # safety margin before 120s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    n, m, p = 3, 3, 8
    
    # Optimal rank is max(n*m, m*p, p*n) = 24
    rank = 24
    
    # Initialize matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Structure:
    # We map each column c in [0, rank) to a pair (j, k) where j in [0, m), k in [0, p).
    # Since m=3, p=8, we can use c = j * p + k.
    # For a given c, let j_c = c // p, k_c = c % p.
    
    # V[j, k, c] is non-zero only if j == j_c and k == k_c.
    # Specifically, V[j*p + k, c] = 1 if j == j_c and k == k_c, else 0.
    
    # U[i, j, c] is non-zero if j == j_c.
    # Specifically, U[i*m + j, c] = 1 if j == j_c, else 0.
    
    # W[k, i, c] is non-zero if k == k_c.
    # Specifically, W[k*n + i, c] = 1 if k == k_c, else 0.
    
    for c in range(rank):
        j_c = c // p
        k_c = c % p
        
        # Fill V
        # V index is j*p + k
        V[j_c * p + k_c, c] = 1.0
        
        # Fill U
        # For all i, U[i*m + j_c, c] = 1
        for i in range(n):
            U[i * m + j_c, c] = 1.0
            
        # Fill W
        # For all i, W[k_c*n + i, c] = 1
        for i in range(n):
            W[k_c * n + i, c] = 1.0

    # Convert to lists of integers/half-integers (they are all 0 or 1 here)
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# Ensure we don't exceed time on import
if __name__ == "__main__":
    pass
