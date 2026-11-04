"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition correctly represents matrix multiplication."""
    T_expected = np.ones((n, m, p))
    T_computed = np.zeros((n, m, p))
    for r in range(U.shape[1]):
        U_r = U[:, r].reshape(n, m)
        V_r = V[:, r].reshape(m, p)
        W_r = W[:, r].reshape(p, n)
        # T[i,j,k] += U_r[i,j] * V_r[j,k] * W_r[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_computed[i, j, k] += U_r[i, j] * V_r[j, k] * W_r[k, i]
    return np.allclose(T_computed, T_expected, atol=tol)

def _alternating_minimization(U, V, W, n, m, p, max_iter=2000, tol=1e-10):
    """Alternating minimization to refine U, V, W."""
    rank = U.shape[1]
    prev_err = np.inf
    
    for it in range(max_iter):
        # Fix V, W, solve for U
        # For each (i,j): sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all k?
        # No, the equation is sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k.
        # This is not linear in U for fixed V,W if we sum over r?
        # Wait, for fixed V, W, the equation for U[i*m+j, :] is:
        # sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all k.
        # Let C_{k,r} = V[j*p+k, r] * W[k*n+i, r].
        # Then sum_r U[i*m+j, r] * C_{k,r} = 1 for all k.
        # This is a linear system: C^T u = 1, where u = U[i*m+j, :].
        
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                C = np.zeros((p, rank))
                for k in range(p):
                    C[k, :] = V[j * p + k, :] * W[k * n + i, :]
                # Solve C^T u = 1
                try:
                    u, residuals, rank_c, sv = np.linalg.lstsq(C.T, np.ones(p), rcond=None)
                    U_new[i * m + j, :] = u
                except np.linalg.LinAlgError:
                    return None, None, None

        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, rank))
                for i in range(n):
                    B[i, :] = U_new[i * m + j, :] * W[k * n + i, :]
                try:
                    v, residuals, rank_b, sv = np.linalg.lstsq(B.T, np.ones(n), rcond=None)
                    V_new[j * p + k, :] = v
                except np.linalg.LinAlgError:
                    return None, None, None

        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U_new[i * m + j, :] * V_new[j * p + k, :]
                try:
                    w, residuals, rank_a, sv = np.linalg.lstsq(A.T, np.ones(m), rcond=None)
                    W_new[k * n + i, :] = w
                except np.linalg.LinAlgError:
                    return None, None, None

        U, V, W = U_new, V_new, W_new
        
        # Check convergence
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
            return U, V, W
            
    if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
        return U, V, W
    return None, None, None

def _block_init(n, m, p, rank, seed=42):
    """Initialize U, V, W with block structure for 3+3+2 split of p=8."""
    rng = np.random.RandomState(seed)
    
    # Split p=8 into p1=3, p2=3, p3=2
    p1, p2, p3 = 3, 3, 2
    r1, r2, r3 = 23, 23, 18 # Target ranks for each block
    if r1 + r2 + r3 > rank:
        # Adjust if rank is too small
        r1 = rank // 3
        r2 = rank // 3
        r3 = rank - r1 - r2
        
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Block 1: p in [0, 3)
    # U is same for all blocks? No, U depends on i,j.
    # In the decomposition, the rank components are independent.
    # So we can assign columns of U, V, W to each block.
    
    # Block 1 columns: 0 to r1-1
    U[:, 0:r1] = rng.randn(n * m, r1)
    # V for block 1: only k in [0, 3) should be non-zero?
    # V is m x p. For block 1, we only care about k in [0, 3).
    V[:, 0:r1] = 0
    V[0:m*1, 0:r1] = rng.randn(m, r1) # j=0, k=0..2? No, V is indexed by j*p+k.
    # V[j, k] corresponds to row j*p+k in V.
    # For block 1 (k=0,1,2), we set V[j, 0:3] to random.
    for j in range(m):
        V[j * p + 0 : j * p + 3, 0:r1] = rng.randn(3, r1)
    # W for block 1: W[k, i] for k in [0, 3).
    for k in range(3):
        W[k * n : (k+1) * n, 0:r1] = rng.randn(n, r1)
        
    # Block 2 columns: r1 to r1+r2-1
    offset2 = r1
    U[:, offset2:offset2+r2] = rng.randn(n * m, r2)
    V[:, offset2:offset2+r2] = 0
    for j in range(m):
        V[j * p + 3 : j * p + 6, offset2:offset2+r2] = rng.randn(3, r2)
    for k in range(3, 6):
        W[k * n : (k+1) * n, offset2:offset2+r2] = rng.randn(n, r2)
        
    # Block 3 columns: r1+r2 to rank-1
    offset3 = r1 + r2
    U[:, offset3:rank] = rng.randn(n * m, r3)
    V[:, offset3:rank] = 0
    for j in range(m):
        V[j * p + 6 : j * p + 8, offset3:rank] = rng.randn(2, r3)
    for k in range(6, 8):
        W[k * n : (k+1) * n, offset3:rank] = rng.randn(n, r3)
        
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Start with schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Target rank 64 (23+23+18)
    for target_rank in [64, 63, 62, 61, 60]:
        if time.time() > deadline - 30:
            break
            
        for attempt in range(5):
            if time.time() > deadline - 30:
                break
                
            U, V, W = _block_init(n, m, p, target_rank, seed=42 + target_rank * 10 + attempt)
            
            U_res, V_res, W_res = _alternating_minimization(U, V, W, n, m, p, max_iter=500)
            
            if U_res is not None and _verify_decomposition(U_res, V_res, W_res, n, m, p):
                best_rank = target_rank
                best_result = {"U": U_res.tolist(), "V": V_res.tolist(), "W": W_res.tolist()}
                break
        
        if best_rank == target_rank:
            break
            
    return best_result
# EVOLVE-BLOCK-END
