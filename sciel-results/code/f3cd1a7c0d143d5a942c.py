import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Create index arrays for the rank dimension
    c = np.arange(rank)
    
    # Decompose linear index c into (i, j, k)
    # c = i * (m * p) + j * p + k
    i_idx = c // (m * p)
    j_idx = (c // p) % m
    k_idx = c % p
    
    # Calculate row indices for U, V, W
    # U shape: (n*m, rank) -> row = i * m + j
    # V shape: (m*p, rank) -> row = j * p + k
    # W shape: (p*n, rank) -> row = k * n + i
    
    u_rows = i_idx * m + j_idx
    v_rows = j_idx * p + k_idx
    w_rows = k_idx * n + i_idx
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Set the 1s
    U[u_rows, c] = 1
    V[v_rows, c] = 1
    W[w_rows, c] = 1
    
    # Convert to lists of lists as per interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
