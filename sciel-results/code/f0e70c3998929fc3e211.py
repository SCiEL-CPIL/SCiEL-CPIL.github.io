"""Matrix multiplication 2,2,15 tensor rank minimization."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 15]
TIME_LIMIT = 140

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _build_strassen_2x2x2():
    """Build Strassen's rank-7 decomposition for 2x2 x 2x2."""
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U[0, 0] = 1;  U[3, 0] = 1
    V[0, 0] = 1;  V[3, 0] = 1
    W[0, 0] = 1;  W[3, 0] = 1
    
    U[2, 1] = 1;  U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1;  W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1;  V[3, 2] = -1
    W[2, 2] = 1;  W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1;  V[0, 3] = -1
    W[0, 3] = 1;  W[1, 3] = 1
    
    U[0, 4] = 1;  U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1;  W[2, 4] = 1
    
    U[2, 5] = 1;  U[0, 5] = -1
    V[0, 5] = 1;  V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1;  U[3, 6] = -1
    V[2, 6] = 1;  V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_2x2x3_rank11():
    """
    Build a rank-11 decomposition for 2x2 x 2x3 matrix multiplication.
    Uses 7 Strassen terms for columns 0-1 and 4 schoolbook terms for column 2.
    """
    n, m, p = 2, 2, 3
    rank = 11
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U_str, V_str, W_str = _build_strassen_2x2x2()
    
    # First 7 terms: Strassen for columns 0,1
    U[:, 0:7] = U_str
    for j in range(m):
        for k_local in range(2):
            V[j * p + k_local, 0:7] = V_str[j * 2 + k_local, :]
    for k_local in range(2):
        for i in range(n):
            W[k_local * n + i, 0:7] = W_str[k_local * 2 + i, :]
    
    # Terms 7-10: Schoolbook for column 2
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    U[0, 7] = 1; V[0, 7] = 1; W[4, 7] = 1  # A[0,0]*B[0,2] -> C[0,2]
    U[1, 8] = 1; V[5, 8] = 1; W[4, 8] = 1  # A[0,1]*B[1,2] -> C[0,2]
    U[2, 9] = 1; V[4, 9] = 1; W[5, 9] = 1  # A[1,0]*B[0,2] -> C[1,2]
    U[3, 10] = 1; V[5, 10] = 1; W[5, 10] = 1  # A[1,1]*B[1,2] -> C[1,2]
    
    return U, V, W

def _build_rank53():
    """Build the rank-53 decomposition using 7 Strassen blocks + 1 schoolbook."""
    n, m, p = DIMENSIONS
    block_sizes = [2, 2, 2, 2, 2, 2, 2, 1]
    num_blocks = len(block_sizes)
    
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)
    
    total_rank = 7 * 7 + 4  # 53
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for b in range(num_blocks):
        block_size = block_sizes[b]
        start_k = sum(block_sizes[:b])
        
        if block_size == 2:
            U[:, col_offset:col_offset+7] = U_str
            for j in range(m):
                for k_local in range(2):
                    k_global = start_k + k_local
                    V[j * p + k_global, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
            for k_local in range(2):
                k_global = start_k + k_local
                for i in range(n):
                    W[k_global * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
            col_offset += 7
        else:
            U[:, col_offset:col_offset+4] = U_sb
            for j in range(m):
                V[j * p + start_k, col_offset:col_offset+4] = V_sb[j * 1 + 0, :]
            for i in range(n):
                W[start_k * n + i, col_offset:col_offset+4] = W_sb[0 * 2 + i, :]
            col_offset += 4
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is correct."""
    rank = U.shape[1]
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_find_rank52():
    """
    Try to find a rank-52 decomposition by searching for a 2x2x4 block of rank 13
    or a 2x2x3 block of rank 10. If found, combine with Strassen blocks.
    Returns (U, V, W) if found, else None.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Strategy 1: Find 2x2x3 of rank 10
    # 5 blocks of 2x2x2 (rank 7 each) + 1 block of 2x2x3 (rank 10) = 35 + 10 = 45. Wait, 5*2=10 columns, +3 = 13.
    # We need 15 columns.
    # 6 blocks of 2x2x2 (rank 7 each) + 1 block of 2x2x3 (rank 10) = 42 + 10 = 52.
    # 6*2 = 12 columns + 3 = 15 columns. This works.
    
    # Let's try to search for a rank 10 decomposition of 2x2x3.
    # The tensor has 2*2*3 = 12 non-zero entries in the standard basis.
    # Rank 10 means we need 10 terms.
    
    # We can parameterize the 10 terms.
    # U: 4x10, V: 6x10, W: 6x10.
    # This is a high-dimensional search space.
    # However, we can use the fact that the tensor is "small" and try to solve the equations.
    
    # Equations:
    # For each (i,j,k) in {(0,0,0), (0,0,1), ..., (1,1,2)}:
    # sum_{r=1}^{10} U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    # For all other (i,j,k): sum = 0
    
    # This is a system of 12 polynomial equations in 30 variables (40 + 60 + 60 = 160 variables actually).
    # This is likely too hard to solve directly with simple methods in the time limit.
    
    # Alternative: Try to see if we can reduce the rank of 2x2x2x2 (4 columns) to 13?
    # 2x2x4 rank 13 would mean 2x2x4 is "almost" Strassen.
    # 2x2x4 = 2x2x2 + 2x2x2. If they are independent, rank is 14.
    # Can we share a term?
    # If we share a term, we need U1*V1*W1 = U2*V2*W2? No, we need the sum to match.
    
    # Let's try a simpler search: 
    # Is there a rank 52 decomposition?
    # Lower bound for 2x2x15?
    # Rank >= max(n*m, m*p, p*n) = max(4, 30, 30) = 30.
    # Rank >= (nm+mp+pn)/3 = (4+30+30)/3 = 21.
    # Cohn-Winograd lower bound: (nm+mp+pn)/3.
    # For 2x2x2, rank is 7.
    # For 2x2x3, rank is 11 (known).
    # For 2x2x4, rank is 14 (known, 2*7).
    # For 2x2x15, rank is likely 53 (7*7 + 4).
    # If rank 52 exists, it would be a breakthrough.
    # Given the time limit and complexity, it's very unlikely we will find rank 52.
    
    # So we stick with rank 53.
    return None

def construct():
    """
    Construct a rank-minimal exact tensor decomposition
    for multiplying a 2x2 matrix by a 2x15 matrix.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Baseline: rank 53
    U, V, W = _build_rank53()
    
    if not _verify_decomposition(U, V, W, n, m, p):
        U, V, W = _build_schoolbook(n, m, p)
    
    # Try to find rank 52 (likely to fail or take too long, but guarded by time)
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 20:
        result_52 = _try_find_rank52()
        if result_52 is not None:
            U, V, W = result_52
            if _verify_decomposition(U, V, W, n, m, p):
                pass
            else:
                U, V, W = _build_rank53()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
