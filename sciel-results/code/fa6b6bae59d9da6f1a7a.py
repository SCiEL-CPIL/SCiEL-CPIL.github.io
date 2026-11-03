"""Matrix multiplication 3x5x7 - rank-minimal exact tensor decomposition search."""
import numpy as np
from fractions import Fraction
import time
import random
from itertools import product

DIMENSIONS = [3, 5, 7]
DEADLINE = 135.0  # safety margin before 150s kill

def _time_remaining():
    return DEADLINE - time.time()

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)"""
    rank = U.shape[1]
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(val - 1.0) > 1e-6:
                    return False, val
    return True, 1.0

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using Fraction arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                if s != 1:
                    return False
    return True

def _to_rational_list(mat, tol=1e-9):
    """Convert float matrix to list of Fractions, rounding to nearest rational with small denominator."""
    n_rows = len(mat)
    n_cols = len(mat[0])
    result = [[Fraction(0) for _ in range(n_cols)] for _ in range(n_rows)]
    for i in range(n_rows):
        for j in range(n_cols):
            val = mat[i][j]
            if abs(val) < tol:
                result[i][j] = Fraction(0)
            else:
                found = False
                for denom in range(1, 21):
                    num = round(val * denom)
                    if abs(val - num/denom) < tol:
                        result[i][j] = Fraction(num, denom)
                        found = True
                        break
                if not found:
                    result[i][j] = Fraction(val).limit_denominator(1000)
    return result

def _local_search(U, V, W, n, m, p, max_iters=5000, tol=1e-10):
    """Local refinement to minimize max |sum_r U*V*W - 1| over all (i,j,k)."""
    rank = U.shape[1]
    U = U.astype(np.float64).copy()
    V = V.astype(np.float64).copy()
    W = W.astype(np.float64).copy()
    
    def objective():
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    best_obj = objective()
    lr = 0.05
    for it in range(max_iters):
        if _time_remaining() < 5:
            break
        
        # Compute full gradient for each matrix and do a coordinated step
        # This is more effective than coordinate descent
        U_grad = np.zeros_like(U)
        V_grad = np.zeros_like(V)
        W_grad = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err = val - 1.0
                    if abs(err) > 1e-12:
                        U_grad[i*m+j, :] += err * V[j*p+k, :] * W[k*n+i, :]
                        V_grad[j*p+k, :] += err * U[i*m+j, :] * W[k*n+i, :]
                        W_grad[k*n+i, :] += err * U[i*m+j, :] * V[j*p+k, :]
        
        # Take a step
        U_new = U - lr * U_grad
        V_new = V - lr * V_grad
        W_new = W - lr * W_grad
        
        obj_new = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_new[i*m+j, :] * V_new[j*p+k, :] * W_new[k*n+i, :])
                    err = abs(val - 1.0)
                    if err > obj_new:
                        obj_new = err
        
        if obj_new < best_obj:
            U, V, W = U_new, V_new, W_new
            best_obj = obj_new
            # Decay learning rate
            if it % 200 == 0 and it > 0:
                lr *= 0.85
        else:
            # If no improvement, reduce learning rate
            lr *= 0.7
    
    return U, V, W, best_obj

def _random_decomposition(rank, n, m, p, seed=None):
    """Generate a random rank-r decomposition."""
    if seed is not None:
        random.seed(seed)
    U = np.random.randn(n*m, rank) * 0.3
    V = np.random.randn(m*p, rank) * 0.3
    W = np.random.randn(p*n, rank) * 0.3
    return U, V, W

def _try_rank(rank, n, m, p, max_time=20.0):
    """Try to find a valid decomposition of given rank."""
    start = time.time()
    best_obj = float('inf')
    best_U = best_V = best_W = None
    
    num_attempts = 30
    for attempt in range(num_attempts):
        if time.time() - start > max_time - 2:
            break
        if _time_remaining() < 10:
            break
        
        U, V, W = _random_decomposition(rank, n, m, p, seed=attempt*1000+42)
        
        # Local search
        U, V, W, obj = _local_search(U, V, W, n, m, p, max_iters=3000)
        
        if obj < best_obj:
            best_obj = obj
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Check if we got close enough to try rational recovery
        if obj < 1e-4:
            U_r = _to_rational_list(best_U.tolist())
            V_r = _to_rational_list(best_V.tolist())
            W_r = _to_rational_list(best_W.tolist())
            
            if _verify_exact(U_r, V_r, W_r, n, m, p):
                return U_r, V_r, W_r
    
    return None

def _block_decomposition(n, m, p, block_sizes, max_time=30.0):
    """
    Decompose the multiplication tensor by splitting dimensions into blocks.
    For each block of (n_i, m_j, p_k), compute a schoolbook decomposition
    and combine them. The rank is sum of n_i * m_j * p_k over all blocks.
    This can be less than n*m*p if the blocks are structured.
    """
    start = time.time()
    total_rank = 0
    for (n_i, m_j, p_k) in block_sizes:
        total_rank += n_i * m_j * p_k
    
    if total_rank >= n * m * p:
        return None
    
    # Build the combined decomposition
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for (n_i, m_j, p_k) in block_sizes:
        for i in range(n_i):
            for j in range(m_j):
                for k in range(p_k):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
    
    # Verify
    if _verify_exact(U, V, W, n, m, p):
        return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 105
    
    # First, ensure we have a valid fallback (schoolbook)
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    fallback = schoolbook()
    
    # Try block decompositions that split dimensions
    # Split p=7 into 3+4: rank = 3*5*3 + 3*5*4 = 45 + 60 = 105 (same)
    # Split m=5 into 2+3: rank = 3*2*7 + 3*3*7 = 42 + 63 = 105 (same)
    # Split n=3 into 1+2: rank = 1*5*7 + 2*5*7 = 35 + 70 = 105 (same)
    # These don't help with simple splitting.
    
    # Try a more sophisticated approach: use the structure of the problem
    # The key insight is that for rectangular matrix multiplication,
    # there are known lower-rank decompositions for small cases.
    
    # Try random search with local refinement for ranks below schoolbook
    lower_bound = max(n*m, m*p, p*n)  # 35
    
    # Try a range of ranks, starting from just above the lower bound
    ranks_to_try = list(range(lower_bound + 5, schoolbook_rank, 3))
    # Make sure we include some key values
    for r in [35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105]:
        if r not in ranks_to_try and lower_bound <= r <= schoolbook_rank:
            ranks_to_try.append(r)
    ranks_to_try = sorted(set(ranks_to_try))
    
    best_result = fallback
    best_rank = schoolbook_rank
    
    for rank in ranks_to_try:
        if _time_remaining() < 15:
            break
        if rank >= best_rank:
            break
            
        result = _try_rank(rank, n, m, p, max_time=min(15.0, _time_remaining() - 5))
        if result is not None:
            U_r, V_r, W_r = result
            best_result = {"U": U_r, "V": V_r, "W": W_r}
            best_rank = rank
            # If we found a valid decomposition, we can stop (we want minimum rank)
            if rank == lower_bound:
                break
    
    return best_result
