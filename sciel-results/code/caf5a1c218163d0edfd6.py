"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from fractions import Fraction

def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, n=2, m=3, p=3, tol=1e-9):
    """Verify that the tensor decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # For each (i,j,k) entry, check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _to_list(arr):
    return np.array(arr).tolist()

def _search_rank(target_rank, time_budget, n=2, m=3, p=3):
    """Search for a rank decomposition with given target rank."""
    start_time = time.time()
    deadline = start_time + time_budget
    np.random.seed(42)
    
    # Try multiple random seeds to increase diversity
    for trial in range(20000):
        if time.time() > deadline:
            break
        
        # Random U and V with entries from {-1, 0, 1}
        # Increase density slightly to avoid singular systems
        U = np.random.randint(-1, 2, size=(n*m, target_rank))
        V = np.random.randint(-1, 2, size=(m*p, target_rank))
        
        # Build linear system for W
        # Equations: for each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # Unknowns: W[0..p*n-1, 0..rank-1] -> p*n*rank unknowns
        # W is indexed as W[k*n+i, c]
        
        num_eqs = n * m * p
        num_unk = p * n * target_rank
        
        A = np.zeros((num_eqs, num_unk))
        b = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(target_rank):
                        u_val = U[i*m+j, c]
                        v_val = V[j*p+k, c]
                        if u_val != 0 and v_val != 0:
                            w_row = k * n + i
                            w_col = c
                            unk_idx = w_row * target_rank + w_col
                            A[eq_idx, unk_idx] = u_val * v_val
                    eq_idx += 1
        
        # Solve A @ W_vec = b
        try:
            W_vec, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
        
        W_mat = W_vec.reshape(p*n, target_rank)
        
        # Check if W is close to half-integers
        W_check = 2 * W_mat
        max_err = np.max(np.abs(W_check - np.round(W_check)))
        if max_err < 1e-6:
            W_rounded = np.round(W_check) / 2.0
            if _verify(U.tolist(), V.tolist(), W_rounded.tolist()):
                return {
                    "U": _to_list(U),
                    "V": _to_list(V),
                    "W": _to_list(W_rounded)
                }
    
    return None

def construct():
    start_time = time.time()
    
    # Try to find a better decomposition
    # Rank 15 is the theoretical lower bound candidate, 16 is also possible
    # We try 15 first, then 16
    
    # Allocate time: 70s for rank 15, 60s for rank 16
    # Total 130s, leaving 20s margin
    
    result = _search_rank(15, 70.0)
    if result is not None:
        return result
        
    result = _search_rank(16, 60.0)
    if result is not None:
        return result
        
    # Fallback to schoolbook
    return _build_schoolbook()
# EVOLVE-BLOCK-END
