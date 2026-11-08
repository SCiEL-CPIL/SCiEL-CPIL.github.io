import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

# Precomputed rank-49 Strassen decomposition (2-level)
# This is the known optimal rank for 4x4 matrix multiplication.
# The decomposition uses a Kronecker product structure:
# U[flat, K*7+l] = S_A[K, block_idx] * S_A[l, local_idx]
# V[flat, K*7+l] = S_B[K, block_idx] * S_B[l, local_idx]  
# W[flat, K*7+l] = S_C[K, block_idx] * S_C[l, local_idx]

def _build_decomposition():
    """Build the rank-49 Strassen decomposition for 4x4 matrix multiplication."""
    n, m, p = 4, 4, 4
    rank = 49
    
    # Strassen coefficient matrices (7x4) for 2x2 blocks
    # Row order: M0..M6, Col order: element0..element3
    # For A: cols = [A00, A01, A10, A11]
    # For B: cols = [B00, B01, B10, B11]
    # For C: cols = [C00, C01, C10, C11]
    
    S_A = np.array([
        [1, 0, 0, 1],    # M0: A00 + A11
        [0, 0, 1, 1],    # M1: A10 + A11
        [1, 0, 0, 0],    # M2: A00
        [0, 0, 0, 1],    # M3: A11
        [1, 1, 0, 0],    # M4: A00 + A10
        [-1, 0, 1, 0],   # M5: A10 - A00 (adjusted for this decomposition variant)
        [0, 1, 0, -1]    # M6: A01 - A11
    ], dtype=int)
    
    S_B = np.array([
        [1, 0, 0, 1],    # M0: B00 + B11
        [1, 0, 0, 0],    # M1: B00
        [0, 1, 0, -1],   # M2: B01 - B11
        [-1, 0, 1, 0],   # M3: B10 - B00
        [0, 0, 0, 1],    # M4: B11
        [1, 1, 0, 0],    # M5: B00 + B01
        [0, 0, 1, 1]     # M6: B10 + B11
    ], dtype=int)
    
    S_C = np.array([
        [1, 0, 0, 1],    # M0: C00 + C11
        [0, 0, 1, -1],   # M1: C10 - C11
        [0, 1, 0, 1],    # M2: C01 + C11
        [1, 0, 1, 0],    # M3: C00 + C10
        [-1, 1, 0, 0],   # M4: C01 - C00
        [0, 0, 0, 1],    # M5: C11
        [1, 0, 0, 0]     # M6: C00
    ], dtype=int)
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Precompute index mappings
    # For U: A[i,j], flat = i*m+j, block = (i//2)*2 + (j//2), local = (i%2)*2 + (j%2)
    idx_u = np.arange(n * m)
    i_u = idx_u // m
    j_u = idx_u % m
    block_u = (i_u // 2) * 2 + (j_u // 2)
    local_u = (i_u % 2) * 2 + (j_u % 2)
    
    # For V: B[j,k], flat = j*p+k, block = (j//2)*2 + (k//2), local = (j%2)*2 + (k%2)
    idx_v = np.arange(m * p)
    j_v = idx_v // p
    k_v = idx_v % p
    block_v = (j_v // 2) * 2 + (k_v // 2)
    local_v = (j_v % 2) * 2 + (k_v % 2)
    
    # For W: trace convention W[k*n+i], flat = k*n+i, block = (i//2)*2 + (k//2), local = (i%2)*2 + (k%2)
    idx_w = np.arange(p * n)
    k_w = idx_w // n
    i_w = idx_w % n
    block_w = (i_w // 2) * 2 + (k_w // 2)
    local_w = (i_w % 2) * 2 + (k_w % 2)
    
    for K in range(7):
        for l in range(7):
            r = K * 7 + l
            U[:, r] = S_A[K, block_u] * S_A[l, local_u]
            V[:, r] = S_B[K, block_v] * S_B[l, local_v]
            W[:, r] = S_C[K, block_w] * S_C[l, local_w]
    
    return U, V, W

# Cache the precomputed decomposition
_U, _V, _W = _build_decomposition()

def construct():
    return {"U": _U.tolist(), "V": _V.tolist(), "W": _W.tolist()}

# EVOLVE-BLOCK-END
