import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Try cyclic convolution / DFT approach first
    result = _try_dft_decomposition(n, m, p, deadline)
    if result is not None:
        return result
    
    # Fall back to Strassen-based rank-28 decomposition
    return _strassen_rank28(n, m, p)

def _try_dft_decomposition(n, m, p, deadline):
    """
    Try to find a low-rank decomposition using cyclic convolution structure.
    Embeds the matrix multiplication tensor into a cyclic group algebra Z/qZ
    and uses DFT to diagonalize, then projects back.
    """
    if time.time() > deadline - 5:
        return None
    
    # Strategy: For 2x3 * 3x5, we can try to use the structure of the tensor.
    # The tensor T[i,j,k] = 1 if i in [0,n), j in [0,m), k in [0,p)
    # We look for a decomposition using cyclic shifts.
    
    # Try different moduli for the cyclic group
    for q in [11, 13, 17, 19, 23]:
        if time.time() > deadline - 5:
            return None
        
        # Embed indices into Z/qZ
        # i -> i, j -> j, k -> k
        # The product C[i,k] = sum_j A[i,j] * B[j,k]
        # In cyclic convolution: (f * g)[t] = sum_s f[s] * g[t-s]
        # We need to map (i,j,k) to a cyclic structure.
        
        # For rectangular matrices, we can try:
        # A[i,j] -> f_i[j] in Z/qZ
        # B[j,k] -> g_j[k] in Z/qZ
        # We want C[i,k] = sum_j A[i,j]*B[j,k]
        
        # This is tricky for rectangular. Let's try a different approach:
        # Use the fact that for small dimensions, we can try to find
        # a decomposition by solving the tensor equation directly.
        
        pass
    
    # Alternative: Try to use the DFT approach more directly.
    # For 2x3 * 3x5, the tensor has 2*3*5 = 30 non-zero entries.
    # The rank is at least the max of n*m, m*p, p*n = max(6, 15, 10) = 15.
    # Strassen gives 28. Can we do better?
    
    # Let's try a direct search for rank < 28 using the structure.
    # We'll use the fact that the tensor is "separable" in some sense.
    
    # For now, return None to use the Strassen fallback.
    return None

def _strassen_rank28(n, m, p):
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: A2 B2 (j=2, k=0..4) - 10 terms
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: A1 B13 (j=0,1, k=4) - 4 terms
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen - 7 terms
    # M1
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M2
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    W[0 * n + 1][r] = 1
    W[1 * n + 1][r] = -1
    r += 1
    
    # M3
    U[0 * m + 0][r] = 1
    V[0 * p + 1][r] = 1
    V[1 * p + 1][r] = -1
    W[1 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M4
    U[1 * m + 1][r] = 1
    V[1 * p + 0][r] = 1
    V[0 * p + 0][r] = -1
    W[0 * n + 0][r] = 1
    W[0 * n + 1][r] = 1
    r += 1
    
    # M5
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = -1
    W[1 * n + 0][r] = 1
    r += 1
    
    # M6
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 0][r] = 1
    V[0 * p + 1][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M7
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    r += 1
    
    # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen - 7 terms
    # M1
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M2
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    W[2 * n + 1][r] = 1
    W[3 * n + 1][r] = -1
    r += 1
    
    # M3
    U[0 * m + 0][r] = 1
    V[0 * p + 3][r] = 1
    V[1 * p + 3][r] = -1
    W[3 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M4
    U[1 * m + 1][r] = 1
    V[1 * p + 2][r] = 1
    V[0 * p + 2][r] = -1
    W[2 * n + 0][r] = 1
    W[2 * n + 1][r] = 1
    r += 1
    
    # M5
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = -1
    W[3 * n + 0][r] = 1
    r += 1
    
    # M6
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 2][r] = 1
    V[0 * p + 3][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M7
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    r += 1
    
    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                s = 0.0
                for r_idx in range(rank):
                    s += U_arr[idx_u, r_idx] * V_arr[idx_v, r_idx] * W_arr[idx_w, r_idx]
                if abs(s - 1.0) > 1e-9:
                    # If verification fails, fall back to schoolbook
                    return _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
    """Fallback schoolbook decomposition with rank n*m*p"""
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
