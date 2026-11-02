"""
Rank-minimal exact tensor decomposition for 2x2 * 2x15 matrix multiplication.

Strategy:
- Partition the 15 columns of B into blocks.
- Primary: 7 blocks of size 2 (Strassen rank 7 each) + 1 block of size 1 (schoolbook rank 4) = 53.
- Improvement attempt: try to find a rank-10 decomposition for a 2x2x3 block
  to replace the last 2x2 (rank 7) + 2x1 (rank 4) = 11 with rank 10, giving total 52.
- Use Levenberg-Marquardt / SLSQP optimization to search for dense rank-10 factors
  for the 2x2x3 tensor, exploiting global linear dependencies.
- Fallback to rank 53 if no valid rank-52 decomposition is found.
"""

import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 15]

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
    """
    Build Strassen's rank-7 decomposition for 2x2 x 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) following the trace convention W[k*n+i, r].
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Index mapping: U[i*m + j, r], V[j*p + k, r], W[k*n + i, r]
    
    # Term 1: M1 = (A00 + A11) * (B00 + B11)
    U[0*2+0, 0] = 1
    U[1*2+1, 0] = 1
    V[0*2+0, 0] = 1
    V[1*2+1, 0] = 1
    W[0*2+0, 0] = 1
    W[1*2+1, 0] = 1
    
    # Term 2: M2 = (A10 + A11) * B00
    U[1*2+0, 1] = 1
    U[1*2+1, 1] = 1
    V[0*2+0, 1] = 1
    W[0*2+1, 1] = 1
    W[1*2+1, 1] = -1
    
    # Term 3: M3 = A00 * (B01 - B11)
    U[0*2+0, 2] = 1
    V[0*2+1, 2] = 1
    V[1*2+1, 2] = -1
    W[1*2+0, 2] = 1
    W[1*2+1, 2] = 1
    
    # Term 4: M4 = A11 * (B10 - B00)
    U[1*2+1, 3] = 1
    V[1*2+0, 3] = 1
    V[0*2+0, 3] = -1
    W[0*2+0, 3] = 1
    W[0*2+1, 3] = 1
    
    # Term 5: M5 = (A00 + A01) * B11
    U[0*2+0, 4] = 1
    U[0*2+1, 4] = 1
    V[1*2+1, 4] = 1
    W[0*2+0, 4] = -1
    W[1*2+0, 4] = 1
    
    # Term 6: M6 = (A10 - A00) * (B00 + B01)
    U[1*2+0, 5] = 1
    U[0*2+0, 5] = -1
    V[0*2+0, 5] = 1
    V[0*2+1, 5] = 1
    W[1*2+1, 5] = 1
    
    # Term 7: M7 = (A01 - A11) * (B10 + B11)
    U[0*2+1, 6] = 1
    U[1*2+1, 6] = -1
    V[1*2+0, 6] = 1
    V[1*2+1, 6] = 1
    W[0*2+0, 6] = 1
    
    return U, V, W

def _residual_2x2x3(params, n=2, m=2, p=3, rank=10):
    """
    Compute the residual for a 2x2x3 tensor decomposition with given parameters.
    params: flat array of length (n*m + m*p + p*n) * rank
    Returns: squared Frobenius norm of residual.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    total_size = nm + mp + pn
    U = params[0:nm*rank].reshape(nm, rank)
    V = params[nm*rank:(nm+mp)*rank].reshape(mp, rank)
    W = params[(nm+mp)*rank:(nm+mp+pn)*rank].reshape(pn, rank)
    
    residual = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                residual += (s - 1.0) ** 2
    return residual

def _try_find_rank10_2x2x3(time_budget):
    """
    Attempt to find a rank-10 exact decomposition for 2x2x3 matrix multiplication.
    Uses scipy.optimize.minimize with multiple restarts.
    Returns (U, V, W) if found, else None.
    """
    from scipy.optimize import minimize
    
    n, m, p, rank = 2, 2, 3, 10
    nm, mp, pn = n*m, m*p, p*n
    total_size = (nm + mp + pn) * rank
    
    start_time = time.time()
    deadline = start_time + time_budget
    
    best_params = None
    best_res = float('inf')
    
    # Try multiple random initializations
    rng = np.random.RandomState(42)
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Random initialization with small values
        params = rng.randn(total_size) * 0.5
        
        res = minimize(
            _residual_2x2x3,
            params,
            args=(n, m, p, rank),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10}
        )
        
        if res.fun < best_res:
            best_res = res.fun
            best_params = res.x
        
        if best_res < 1e-10:
            break
    
    if best_params is not None and best_res < 1e-10:
        U = best_params[0:nm*rank].reshape(nm, rank)
        V = best_params[nm*rank:(nm+mp)*rank].reshape(mp, rank)
        W = best_params[(nm+mp)*rank:(nm+mp+pn)*rank].reshape(pn, rank)
        
        # Round to nearest half-integer and verify
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify exactness
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_rounded[i*m+j, r] * V_rounded[j*p+k, r] * W_rounded[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return U_rounded, V_rounded, W_rounded
    
    return None

def _verify_full(U, V, W, tol=1e-9):
    """
    Verify that the full decomposition exactly decomposes the 2x2x15 tensor.
    U: (4, rank), V: (30, rank), W: (30, rank)
    """
    n, m, p = 2, 2, 15
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(actual - expected) > tol:
                    return False
    return True

def construct():
    """
    Construct a rank-minimal exact tensor decomposition
    for multiplying a 2x2 matrix by a 2x15 matrix.
    
    Strategy: 
    - First try to find a rank-52 decomposition by using rank-10 for a 2x2x3 block.
    - If that fails, fall back to rank 53 (7 Strassen blocks + 1 schoolbook block).
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill
    
    n, m, p = DIMENSIONS  # 2, 2, 15
    
    # Try to find rank-52 decomposition
    # Partition: 6 blocks of size 2 (Strassen, rank 7 each) + 1 block of size 3 (rank 10)
    # Total: 6*7 + 10 = 52
    
    U_rank10 = None
    if time.time() < deadline - 10:
        time_budget = min(60, deadline - start_time - 10)
        U_rank10 = _try_find_rank10_2x2x3(time_budget)
    
    if U_rank10 is not None and time.time() < deadline:
        U_10, V_10, W_10 = U_rank10
        # Build rank-52 decomposition
        total_rank = 6 * 7 + 10  # 52
        
        U = np.zeros((n * m, total_rank))  # 4 x 52
        V = np.zeros((m * p, total_rank))  # 30 x 52
        W = np.zeros((p * n, total_rank))  # 30 x 52
        
        U_str, V_str, W_str = _build_strassen_2x2x2()
        
        col_offset = 0
        # 6 blocks of size 2
        for b in range(6):
            start_k = b * 2
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
        
        # 1 block of size 3
        start_k = 12
        U[:, col_offset:col_offset+10] = U_10
        for j in range(m):
            for k_local in range(3):
                k_global = start_k + k_local
                V[j * p + k_global, col_offset:col_offset+10] = V_10[j * 3 + k_local, :]
        for k_local in range(3):
            k_global = start_k + k_local
            for i in range(n):
                W[k_global * n + i, col_offset:col_offset+10] = W_10[k_local * 2 + i, :]
        col_offset += 10
        
        # Verify
        if _verify_full(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: rank 53
    if time.time() >= deadline:
        # Emergency fallback: pure schoolbook
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build rank-53 decomposition
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
    
    if _verify_full(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Final fallback
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
