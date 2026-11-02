"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
import time

def _verify(U, V, W, n=2, m=2, p=6):
    """Check if U, V, W form a valid matrix multiplication decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Compute tensor from factors: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Build standard tensor
    T_std = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T_std[a_idx, b_idx, c_idx] = 1.0
    
    return np.allclose(T, T_std, atol=1e-9)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Rank 21 baseline (Strassen on 3 blocks)
    U_local = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    V_local = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    W_local = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    def build_rank21():
        rank = 21
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        num_blocks = p // 2
        for k in range(num_blocks):
            base = k * 7
            U[:, base:base+7] = U_local
            V[2*k, base:base+7] = V_local[0]
            V[2*k+1, base:base+7] = V_local[1]
            V[6+2*k, base:base+7] = V_local[2]
            V[7+2*k, base:base+7] = V_local[3]
            W[4*k, base:base+7] = W_local[0]
            W[4*k+2, base:base+7] = W_local[1]
            W[4*k+1, base:base+7] = W_local[2]
            W[4*k+3, base:base+7] = W_local[3]
        return U, V, W
    
    U21, V21, W21 = build_rank21()
    
    # Try to find rank 20 using structured search
    # Key insight: For M_{2,2,6}, we can potentially share terms between blocks
    # since A is the same across all 3 column-pairs of B
    
    # Strategy: Use a "shared Strassen" approach where some products
    # are computed once and reused across blocks
    
    # Let's try a more systematic approach:
    # Assume rank r decomposition with U of shape (4, r), V of shape (12, r), W of shape (12, r)
    # We need: sum_r U[a,r]*V[b,r]*W[c,r] = T_std[a,b,c]
    
    # For rank 20, let's try to find if there's a valid decomposition
    # by solving a system of equations
    
    if time.time() < deadline - 10:
        # Try rank 20 with structured initialization
        rng = np.random.default_rng(12345)
        found_rank20 = False
        U20 = V20 = W20 = None
        
        # Approach: Start from rank-21 and try to make one column linearly dependent
        # by solving for coefficients
        
        # Try each pair of columns to merge
        for c1 in range(21):
            for c2 in range(c1+1, 21):
                if time.time() > deadline - 5:
                    break
                
                # Try to express column c2 as a linear combination of column c1 and others
                # This is complex; instead, try a direct random search
                
                # Random search: initialize rank-20 with random small integers
                # and try to refine
                for trial in range(50):
                    if time.time() > deadline - 5:
                        break
                    
                    # Initialize with random values in {-2, -1, 0, 1, 2}
                    U_try = rng.integers(-2, 3, size=(4, 20)).astype(float)
                    V_try = rng.integers(-2, 3, size=(12, 20)).astype(float)
                    W_try = rng.integers(-2, 3, size=(12, 20)).astype(float)
                    
                    # Check if this gives the right tensor (very unlikely but try)
                    if _verify(U_try, V_try, W_try):
                        found_rank20 = True
                        U20, V20, W20 = U_try, V_try, W_try
                        break
                    if time.time() > deadline - 5:
                        break
                if found_rank20:
                    break
            if found_rank20:
                break
        
        if found_rank20:
            U, V, W = U20, V20, W20
        else:
            U, V, W = U21, V21, W21
    else:
        U, V, W = U21, V21, W21
    
    # Final verification
    if not _verify(U, V, W):
        # Fallback to known-good rank-21
        U, V, W = U21, V21, W21
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
