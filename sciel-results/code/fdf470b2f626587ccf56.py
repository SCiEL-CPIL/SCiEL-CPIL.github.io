"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]
import time
import itertools
import random
import numpy as np

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
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

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification."""
    r = U.shape[1]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    T_exp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1.0
    return np.allclose(T, T_exp, atol=tol, rtol=0)

def _rank26_decomposition():
    """
    Construct a rank-26 decomposition for 3x3x3 matrix multiplication.
    Uses Strassen's algorithm for the top-left 2x2 block (7 terms)
    and schoolbook for the remaining 19 terms.
    """
    n, m, p = 3, 3, 3
    rank = 26
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # 1. Add 19 schoolbook terms for indices NOT in {0,1}x{0,1}x{0,1}
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i < 2 and j < 2 and k < 2:
                    continue # Skip these, handled by Strassen
                
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
                
    # 2. Add 7 Strassen terms for the 2x2 block (i,j,k in {0,1})
    # Indices in 2x2 block:
    # A: (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4
    # B: (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4
    # C: (0,0)->0, (0,1)->3, (1,0)->1, (1,1)->4  (Note: W index is k*n+i)
    # C[i,k] -> W[k*3+i]
    # C[0,0] -> W[0]
    # C[0,1] -> W[3]
    # C[1,0] -> W[1]
    # C[1,1] -> W[4]
    
    # Strassen terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M7
    
    # Helper to add a term
    def add_term(u_vals, v_vals, w_vals):
        nonlocal col
        for idx, val in u_vals.items():
            U[idx, col] = val
        for idx, val in v_vals.items():
            V[idx, col] = val
        for idx, val in w_vals.items():
            W[idx, col] = val
        col += 1
        
    # M1
    add_term({0:1, 4:1}, {0:1, 4:1}, {0:1, 4:1})
    # M2
    add_term({3:1, 4:1}, {0:1}, {1:1, 4:1}) # C10 (idx 1), C11 (idx 4)
    # M3
    add_term({0:1}, {3:1, 4:-1}, {3:1, 4:-1}) # C01 (idx 3), C11 (idx 4)
    # M4
    add_term({4:1}, {1:1, 0:-1}, {0:-1, 4:-1}) # C00 (idx 0), C11 (idx 4)
    # M5
    add_term({0:1, 1:1}, {4:1}, {3:1, 4:1}) # C01 (idx 3), C11 (idx 4)
    # M6
    add_term({3:1, 0:-1}, {0:1, 1:1}, {0:1, 1:1}) # C00 (idx 0), C10 (idx 1)
    # M7
    add_term({1:1, 4:-1}, {3:1, 4:1}, {0:1, 3:-1}) # C00 (idx 0), C01 (idx 3)
    
    return U, V, W

def _try_rank23_fast(deadline):
    """
    Attempt a very fast search for rank 23.
    Returns None if not found before deadline.
    """
    n, m, p = 3, 3, 3
    dim_a, dim_b, dim_c = 9, 9, 9
    T = np.zeros((dim_a, dim_b, dim_c))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Simple random restarts with ALS
    # Given the 150s limit and that we have a guaranteed 26,
    # we only spend ~5s here.
    
    U0, V0, W0 = _schoolbook(n, m, p)
    # We need to drop 4 terms.
    # This is a heuristic.
    
    # For simplicity and robustness, we skip the complex ALS here
    # and rely on the rank 26 solution.
    # If we want to try, we can do a few iterations.
    # But the prompt says "Keep the program's entry-point...".
    # The current program's ALS is slow and unreliable.
    # I will implement a simplified ALS check.
    
    rank = 23
    rng = np.random.RandomState(42)
    
    # Try a few random initializations
    for attempt in range(5):
        if time.time() > deadline:
            return None
            
        # Random initialization
        U = rng.randn(dim_a, rank)
        V = rng.randn(dim_b, rank)
        W = rng.randn(dim_c, rank)
        
        # A few ALS steps
        for _ in range(20):
            if time.time() > deadline:
                return None
            # Update U
            M = np.einsum('br,cr->bcr', V, W).reshape(dim_b*dim_c, rank)
            for a in range(dim_a):
                U[a, :] = np.linalg.lstsq(M, T[a,:,:].reshape(-1), rcond=None)[0]
            # Update V
            M = np.einsum('ar,cr->acr', U, W).reshape(dim_a*dim_c, rank)
            for b in range(dim_b):
                V[b, :] = np.linalg.lstsq(M, T[:,b,:].reshape(-1), rcond=None)[0]
            # Update W
            M = np.einsum('ar,br->abr', U, V).reshape(dim_a*dim_b, rank)
            for c in range(dim_c):
                W[c, :] = np.linalg.lstsq(M, T[:,:,c].reshape(-1), rcond=None)[0]
                
        # Check error
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(T_approx, T, atol=1e-8):
            return U, V, W
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # 1. Try Rank 23 (limited time)
    if time.time() < deadline:
        result = _try_rank23_fast(deadline)
        if result is not None:
            U, V, W = result
            if _verify_fast(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    # 2. Fallback to Rank 26 (Guaranteed)
    U, V, W = _rank26_decomposition()
    
    # Verify Rank 26
    if _verify_fast(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # 3. Fallback to Schoolbook Rank 27
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
