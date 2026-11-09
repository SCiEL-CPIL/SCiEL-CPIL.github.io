import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

def _elapsed(start_time):
    return time.time() - start_time

def _deadline_exceeded(start_time, deadline=130):
    return _elapsed(start_time) > deadline

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
    return U, V, W

def _verify(U, V, W, n, m, p, tolerance=1e-9):
    """Verify that the decomposition is correct."""
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # Check dimensions
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    
    # For each (i, j, k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # Actually, the tensor decomposition means:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = 1 if this is a schoolbook term, else 0
    # Wait, let me re-read the problem.
    
    # The matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k (it's the full tensor)
    # No wait, that's not right either. Let me think about this more carefully.
    
    # For matrix multiplication C = AB where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has entries T[i,j,k] such that C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i, j, k.
    
    # The decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # Wait, the indices need to be consistent. Let me use the convention from the code.
    
    # U has shape (n*m, rank), so U[i*m+j, r]
    # V has shape (m*p, rank), so V[j*p+k, r]  
    # W has shape (p*n, rank), so W[k*n+i, r]
    
    # The tensor entry T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > tolerance:
                    return False
    return True

def _random_half_integer():
    """Generate a random half-integer: 0, ±0.5, ±1, ±1.5, etc."""
    choices = [0, 0.5, -0.5, 1, -1, 1.5, -1.5, 2, -2]
    return random.choice(choices)

def _local_search(U, V, W, n, m, p, start_time, max_iter=10000):
    """Perform local search to improve the decomposition."""
    rank = U.shape[1]
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    best_rank = rank
    
    for iteration in range(max_iter):
        if _deadline_exceeded(start_time):
            break
        
        # Try to remove a random column if the decomposition still works
        r = random.randint(0, rank - 1)
        
        # Try removing column r
        U_test = np.delete(best_U, r, axis=1)
        V_test = np.delete(best_V, r, axis=1)
        W_test = np.delete(best_W, r, axis=1)
        
        if _verify(U_test, V_test, W_test, n, m, p):
            best_U = U_test
            best_V = V_test
            best_W = W_test
            best_rank -= 1
        
        # Try modifying a random entry
        mat = random.choice(['U', 'V', 'W'])
        if mat == 'U':
            row = random.randint(0, n * m - 1)
            col = random.randint(0, best_rank - 1)
            old_val = best_U[row, col]
            best_U[row, col] = _random_half_integer()
            if not _verify(best_U, best_V, best_W, n, m, p):
                best_U[row, col] = old_val
        elif mat == 'V':
            row = random.randint(0, m * p - 1)
            col = random.randint(0, best_rank - 1)
            old_val = best_V[row, col]
            best_V[row, col] = _random_half_integer()
            if not _verify(best_V, best_V, best_W, n, m, p):
                best_V[row, col] = old_val
        else:
            row = random.randint(0, p * n - 1)
            col = random.randint(0, best_rank - 1)
            old_val = best_W[row, col]
            best_W[row, col] = _random_half_integer()
            if not _verify(best_U, best_V, best_W, n, m, p):
                best_W[row, col] = old_val
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = 130  # safety margin before 150s kill
    
    # Start with schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to improve using Strassen on 2x2x2 blocks
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2: i in {2,3}, j in {2,3}, k in {2,3}
    
    U_strassen = np.zeros((2, 2, 7), dtype=int)
    U_strassen[0, 0, 0] = 1; U_strassen[1, 1, 0] = 1
    U_strassen[1, 0, 1] = 1; U_strassen[1, 1, 1] = 1
    U_strassen[0, 0, 2] = 1
    U_strassen[1, 1, 3] = 1
    U_strassen[0, 0, 4] = 1; U_strassen[0, 1, 4] = 1
    U_strassen[1, 0, 5] = 1; U_strassen[0, 0, 5] = -1
    U_strassen[0, 1, 6] = 1; U_strassen[1, 1, 6] = -1
    
    V_strassen = np.zeros((2, 2, 7), dtype=int)
    V_strassen[0, 0, 0] = 1; V_strassen[1, 1, 0] = 1
    V_strassen[0, 0, 1] = 1
    V_strassen[0, 1, 2] = 1; V_strassen[1, 1, 2] = -1
    V_strassen[1, 0, 3] = 1; V_strassen[0, 0, 3] = -1
    V_strassen[1, 1, 4] = 1
    V_strassen[0, 0, 5] = 1; V_strassen[0, 1, 5] = 1
    V_strassen[1, 0, 6] = 1; V_strassen[1, 1, 6] = 1
    
    W_strassen = np.zeros((2, 2, 7), dtype=int)
    W_strassen[0, 0, 0] = 1; W_strassen[0, 0, 3] = 1; W_strassen[0, 0, 4] = -1; W_strassen[0, 0, 6] = 1
    W_strassen[0, 1, 2] = 1; W_strassen[0, 1, 4] = 1
    W_strassen[1, 0, 1] = 1; W_strassen[1, 0, 3] = 1
    W_strassen[1, 1, 0] = 1; W_strassen[1, 1, 1] = -1; W_strassen[1, 1, 2] = 1; W_strassen[1, 1, 5] = 1
    
    blocks = [
        (0, 2, 0, 2, 0, 2),
        (2, 4, 2, 4, 2, 4)
    ]
    
    # Build Strassen-based decomposition
    total_rank = n * m * p - 2 * 8 + 2 * 7  # 120 - 16 + 14 = 118
    
    U_new = np.zeros((n * m, total_rank))
    V_new = np.zeros((m * p, total_rank))
    W_new = np.zeros((p * n, total_rank))
    
    current_col = 0
    
    strassen_indices = set()
    for (i_s, i_e, j_s, j_e, k_s, k_e) in blocks:
        for i in range(i_s, i_e):
            for j in range(j_s, j_e):
                for k in range(k_s, k_e):
                    strassen_indices.add((i, j, k))
    
    # Add schoolbook terms for non-Strassen indices
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in strassen_indices:
                    U_new[i * m + j, current_col] = 1
                    V_new[j * p + k, current_col] = 1
                    W_new[k * n + i, current_col] = 1
                    current_col += 1
    
    # Add Strassen terms
    for (i_s, i_e, j_s, j_e, k_s, k_e) in blocks:
        for r in range(7):
            for i_loc in range(2):
                for j_loc in range(2):
                    val = U_strassen[i_loc, j_loc, r]
                    if val != 0:
                        i_glob = i_s + i_loc
                        j_glob = j_s + j_loc
                        U_new[i_glob * m + j_glob, current_col] += val
            
            for j_loc in range(2):
                for k_loc in range(2):
                    val = V_strassen[j_loc, k_loc, r]
                    if val != 0:
                        j_glob = j_s + j_loc
                        k_glob = k_s + k_loc
                        V_new[j_glob * p + k_glob, current_col] += val
            
            for i_loc in range(2):
                for k_loc in range(2):
                    val = W_strassen[i_loc, k_loc, r]
                    if val != 0:
                        i_glob = i_s + i_loc
                        k_glob = k_s + k_loc
                        W_new[k_glob * n + i_glob, current_col] += val
            
            current_col += 1
    
    # Verify the Strassen-based decomposition
    if _verify(U_new, V_new, W_new, n, m, p):
        best_U, best_V, best_W = U_new, V_new, W_new
        best_rank = total_rank
    
    # Try local search to further improve
    if not _deadline_exceeded(start_time):
        U_ls, V_ls, W_ls, rank_ls = _local_search(best_U, best_V, best_W, n, m, p, start_time, max_iter=5000)
        if rank_ls < best_rank:
            best_U, best_V, best_W, best_rank = U_ls, V_ls, W_ls, rank_ls
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
