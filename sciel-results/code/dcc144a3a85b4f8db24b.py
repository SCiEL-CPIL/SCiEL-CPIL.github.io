"""Optimized matrix multiplication tensor decomposition for 2x3 by 3x6."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    """Build the trivial rank-nmp schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i*m*p + j*p + k
                U[i*m + j, r] = 1
                V[j*p + k, r] = 1
                W[k*n + i, r] = 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify exact tensor decomposition: sum_r U_{ij,r} V_{jk,r} W_{ki,r} == delta_{j,l}
    for all (i,j,k) and check off-diagonals are zero."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    rank = U.shape[1]
    # Check diagonal terms: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    # Check off-diagonal terms: for j != l, sum_r U[i*m+j, r] * V[l*p+k, r] * W[k*n+i, r] == 0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for l in range(m):
                    if j == l:
                        continue
                    s = np.sum(U[i*m+j] * V[l*p+k] * W[k*n+i])
                    if abs(s) > tol:
                        return False
    return True

def _try_rank(r, n, m, p, deadline, seed_offset=0, max_trials=50):
    """Try to find a valid decomposition of rank r using random search with local refinement."""
    np.random.seed(42 + seed_offset)
    
    for trial in range(max_trials):
        if time.time() > deadline - 5:
            return None
        
        # Initialize with random values from {-1, 0, 1}
        U = np.random.choice([-1, 0, 1], size=(n*m, r))
        V = np.random.choice([-1, 0, 1], size=(m*p, r))
        W = np.random.choice([-1, 0, 1], size=(p*n, r))
        
        if _verify(U, V, W, n, m, p):
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
        
        # Try local refinement: flip random entries and check
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            
            # Flip a random entry in one of the matrices
            mat = np.random.choice([U, V, W])
            idx = np.random.randint(mat.shape[0])
            col = np.random.randint(r)
            old_val = mat[idx, col]
            new_val = np.random.choice([-1, 0, 1])
            if new_val == old_val:
                new_val = (new_val + 1) % 3  # cycle to get a different value
            
            mat[idx, col] = new_val
            if _verify(U, V, W, n, m, p):
                return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
            # Revert
            mat[idx, col] = old_val
    
    return None

def construct():
    """
    Construct a tensor decomposition for multiplying a 2x3 matrix by a 3x6 matrix.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing factor matrices
    that exactly decompose the matrix multiplication tensor.
    """
    start_time = time.time()
    deadline = start_time + 135  # Leave 15 seconds safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Build schoolbook solution as fallback (rank 36)
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    best_result = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Known lower bound: max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    # Try ranks from 18 up to 35
    for r in range(18, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Increase trials for lower ranks since they're harder
        max_trials = 30 if r < 24 else 20
        
        result = _try_rank(r, n, m, p, deadline, seed_offset=r, max_trials=max_trials)
        if result is not None:
            best_rank = r
            best_result = result
            break  # Found minimal rank for this search
    
    return best_result
# EVOLVE-BLOCK-END
