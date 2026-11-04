"""Improved program for matrix multiplication 2x3x4 tensor decomposition."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    deadline = time.time() + 135  # Safety margin before 150s kill
    
    # Schoolbook fallback
    rank_school = n * m * p  # 24
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try ranks from lower bound upward
    # Known lower bound for 2x3x4 is at least 14 (from Strassen-type arguments)
    # Try from 14 to rank_school-1
    for rank in range(14, rank_school):
        if time.time() > deadline:
            break
        
        found = _try_find_solution(n, m, p, rank, deadline=deadline)
        if found is not None:
            U, V, W = found
            best_rank = rank
            best_U, best_V, best_W = U, V, W
            break  # We want minimal rank, so stop at first success
    
    return {"U": best_U, "V": best_V, "W": best_W}

def _try_find_solution(n, m, p, rank, deadline=None, max_iters=500, n_trials=10):
    """Try to find a rank-rank solution using alternating least squares with vectorized ops."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    if deadline is None:
        deadline = time.time() + 135
    
    for trial in range(n_trials):
        if time.time() > deadline:
            return None
        
        seed = 1000 + rank * 1000 + trial
        np.random.seed(seed)
        
        # Initialize with small random values
        U = np.random.randn(nm, rank) * 0.1
        V = np.random.randn(mp, rank) * 0.1
        W = np.random.randn(pn, rank) * 0.1
        
        prev_max_res = float('inf')
        stagnation_count = 0
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U: for each (i,j), solve min ||sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] - 1||
            for i in range(n):
                for j in range(m):
                    # Build matrix A where A[k,t] = V[j*p+k, t] * W[k*n+i, t]
                    A = V[j*p:(j+1)*p, :] * W[i:(i+1)*n, :].T  # shape (p, rank)
                    # Wait, W is indexed as W[k*n+i], so for fixed i, we need W[k*n+i] for k in 0..p-1
                    # W has shape (p*n, rank), so W[k*n+i] is row k*n+i
                    # For fixed i, the rows we need are i, i+n, i+2n, ..., i+(p-1)n
                    # Let's rebuild correctly
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve A @ x = ones(p)
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[i*m+j, :] = x
                    except:
                        pass
            
            # Update V: for each (j,k), solve min ||sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] - 1||
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        V[j*p+k, :] = x
                    except:
                        pass
            
            # Update W: for each (k,i), solve min ||sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] - 1||
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        W[k*n+i, :] = x
                    except:
                        pass
            
            # Check convergence
            max_res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for t in range(rank):
                            s += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                        max_res = max(max_res, abs(s - 1.0))
            
            if max_res < 1e-10:
                break
            
            if max_res > prev_max_res * 0.99:
                stagnation_count += 1
            else:
                stagnation_count = 0
            prev_max_res = max_res
            
            if stagnation_count > 50:
                break
        
        # Try to snap to half-integers or integers
        for scale in [1, 2, 4, 8, 16]:
            U_snapped = np.round(U * scale) / scale
            V_snapped = np.round(V * scale) / scale
            W_snapped = np.round(W * scale) / scale
            if _verify_solution(n, m, p, U_snapped, V_snapped, W_snapped):
                return (U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist())
        
        # Also try just rounding to integers
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        if _verify_solution(n, m, p, U_int, V_int, W_int):
            return (U_int.tolist(), V_int.tolist(), W_int.tolist())
    
    return None

def _verify_solution(n, m, p, U, V, W):
    """Verify that the decomposition is exact."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(rank):
                    s += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                if abs(s - 1.0) > 1e-8:
                    return False
    return True
# EVOLVE-BLOCK-END
