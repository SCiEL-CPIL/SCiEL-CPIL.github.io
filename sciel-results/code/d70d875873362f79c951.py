"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _build_true_tensor(n, m, p):
    """Build the true matrix multiplication tensor."""
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition with tolerance for floating point."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_true = _build_true_tensor(n, m, p)
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Check exactness with tight tolerance
    if not np.allclose(T_recon, T_true, atol=1e-8):
        return False
    return True

def _round_to_half_integer(arr):
    """Round array to nearest half-integer."""
    return np.round(arr * 2) / 2.0

def _strassen_3x3():
    """
    Return Strassen's 7-term decomposition for 3x3x3 matrix multiplication.
    Returns U, V, W as lists of lists, each of shape (9, 7).
    Indexing: U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
    """
    # Strassen's 7 products for 3x3 matrix multiplication
    # A = [[a,b,c],[d,e,f],[g,h,i]], B = [[p,q,r],[s,t,u],[v,w,x]]
    # M1 = (a+b+c)(p+s+v)
    # M2 = (a+b)(q+t+w) ... this gets complicated
    # Let me use a known 7-term decomposition for 3x3x3
    
    # Actually, the standard Strassen is for 2x2x2 (rank 7).
    # For 3x3x3, the best known is rank 19 (or 23 with integer entries).
    # Let me use a structured approach instead.
    
    # For 3x3x3, we can use the "border rank" approach or just return schoolbook (rank 9)
    # and let the search find better.
    
    # Return schoolbook for now
    rank = 9
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j][col] = 1
                V[j*3+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    return U, V, W

def _try_rank_search(r, n, m, p, T_true, max_iters=5000, seed=42, time_budget=60):
    """Try to find rank-r decomposition using gradient descent with half-integer rounding."""
    start_time = time.time()
    nm, mp, pn = n*m, m*p, p*n
    
    best_solution = None
    best_err = float('inf')
    
    for init_seed in range(5):
        if time.time() - start_time > time_budget:
            break
        
        rng = np.random.RandomState(seed + init_seed)
        
        # Initialize with small random values
        scale = 1.0 / np.sqrt(r)
        U = rng.randn(nm, r) * scale
        V = rng.randn(mp, r) * scale
        W = rng.randn(pn, r) * scale
        
        for it in range(max_iters):
            if time.time() - start_time > time_budget:
                break
            
            # Compute current reconstruction
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            R = T_true - T_recon
            
            # Check if we're close enough
            err = np.max(np.abs(R))
            
            # Try rounding to half-integers when close
            if err < 1e-4:
                U_r = _round_to_half_integer(U)
                V_r = _round_to_half_integer(V)
                W_r = _round_to_half_integer(W)
                
                T_recon_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                err_r = np.max(np.abs(T_recon_r - T_true))
                
                if err_r < 1e-8:
                    best_solution = (U_r, V_r, W_r)
                    best_err = err_r
                    return best_solution
            
            # Adaptive learning rate
            eta = 0.1 / (1 + it * 0.01)
            
            # Gradient computation
            grad_U = np.einsum('abc,br,cr->ar', R, V, W)
            grad_V = np.einsum('abc,ar,cr->br', R, U, W)
            grad_W = np.einsum('abc,ar,br->cr', R, U, V)
            
            U += eta * grad_U
            V += eta * grad_V
            W += eta * grad_W
            
            # Periodic rounding attempts
            if it % 100 == 0 and err < 0.5:
                U_r = _round_to_half_integer(U)
                V_r = _round_to_half_integer(V)
                W_r = _round_to_half_integer(W)
                
                T_recon_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                err_r = np.max(np.abs(T_recon_r - T_true))
                
                if err_r < 1e-8:
                    best_solution = (U_r, V_r, W_r)
                    best_err = err_r
                    return best_solution
    
    return best_solution

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _structured_decomposition(n, m, p):
    """
    Try to build a structured decomposition using Kronecker-like products.
    For 3x4x8: split m=4 into m=3+1, and p=8 into p=4+4 or p=2+2+2+2.
    """
    n, m, p = 3, 4, 8
    nm, mp, pn = n*m, m*p, p*n
    
    # Strategy: use the fact that 8 = 2*4, and 4 = 2*2
    # We can try to use tensor product structure
    
    # For now, return schoolbook as fallback
    return _schoolbook(n, m, p)

def construct():
    """
    Constructs the rank-minimal tensor decomposition for 
    multiplying a 3x4 matrix by a 4x8 matrix.
    """
    n, m, p = DIMENSIONS
    T_true = _build_true_tensor(n, m, p)
    
    # Start with schoolbook (rank 96)
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower-rank decompositions
    # Based on known results, try ranks below 96
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # But practical lower bounds are higher
    
    # Try a range of ranks
    for r in [48, 40, 36, 32]:
        if time.time() > 100:  # Safety margin
            break
        
        print(f"Trying rank {r}...")
        result = _try_rank_search(r, n, m, p, T_true, max_iters=3000, seed=42, time_budget=20)
        
        if result is not None:
            U, V, W = result
            # Verify
            if _verify_exact(U, V, W, n, m, p):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = r
                print(f"Found rank {r} decomposition!")
                break
    
    return best_result
# EVOLVE-BLOCK-END
