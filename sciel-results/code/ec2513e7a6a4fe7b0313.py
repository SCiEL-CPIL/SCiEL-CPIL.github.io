"""Combined program for 3x4x4 matrix multiplication decomposition."""
import time
import numpy as np
from scipy import linalg

DIMENSIONS = [3, 4, 4]
_START_TIME = time.time()
_TIMEOUT = 110  # Safety margin

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _try_find_lower_rank(n, m, p, target_rank, max_iter=50):
    """
    Attempt to find a decomposition of rank target_rank using alternating least squares.
    Returns (U, V, W) if successful, else None.
    """
    if time.time() - _START_TIME > _TIMEOUT:
        return None
        
    U_curr = np.random.randn(n * m, target_rank)
    V_curr = np.random.randn(m * p, target_rank)
    W_curr = np.random.randn(p * n, target_rank)
    
    U_curr /= np.sqrt(target_rank)
    V_curr /= np.sqrt(target_rank)
    W_curr /= np.sqrt(target_rank)
    
    for it in range(max_iter):
        if time.time() - _START_TIME > _TIMEOUT:
            return None
            
        # Solve for U
        U_new = np.zeros((n * m, target_rank))
        for i in range(n):
            for j in range(m):
                A_mat = np.zeros((p, target_rank))
                for k in range(p):
                    for r in range(target_rank):
                        A_mat[k, r] = V_curr[j * p + k, r] * W_curr[k * n + i, r]
                b_vec = np.ones(p)
                try:
                    sol, _, _, _ = linalg.lstsq(A_mat, b_vec)
                    U_new[i * m + j, :] = sol
                except:
                    return None
        
        # Solve for V
        V_new = np.zeros((m * p, target_rank))
        for j in range(m):
            for k in range(p):
                A_mat = np.zeros((n, target_rank))
                for i in range(n):
                    for r in range(target_rank):
                        A_mat[i, r] = U_new[i * m + j, r] * W_curr[k * n + i, r]
                b_vec = np.ones(n)
                try:
                    sol, _, _, _ = linalg.lstsq(A_mat, b_vec)
                    V_new[j * p + k, :] = sol
                except:
                    return None
        
        # Solve for W
        W_new = np.zeros((p * n, target_rank))
        for k in range(p):
            for i in range(n):
                A_mat = np.zeros((m, target_rank))
                for j in range(m):
                    for r in range(target_rank):
                        A_mat[j, r] = U_new[i * m + j, r] * V_new[j * p + k, r]
                b_vec = np.ones(m)
                try:
                    sol, _, _, _ = linalg.lstsq(A_mat, b_vec)
                    W_new[k * n + i, :] = sol
                except:
                    return None
        
        U_curr, V_curr, W_curr = U_new, V_new, W_new
        
        # Check convergence/validity
        if _verify(U_curr, V_curr, W_curr, n, m, p):
            return U_curr, V_curr, W_curr
            
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Try to find a lower rank decomposition
    # Start from a reasonable guess, e.g., 19 (known for 4x4x4) or lower
    # For 3x4x4, the rank is likely between 19 and 48.
    # We will try a few target ranks.
    
    target_ranks = [19, 20, 25, 30]
    
    for target_rank in target_ranks:
        if target_rank >= schoolbook_rank:
            continue
            
        result = _try_find_lower_rank(n, m, p, target_rank, max_iter=20)
        if result is not None:
            U, V, W = result
            # Convert to lists for the interface
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
            
    # Fallback to schoolbook
    U, V, W = _schoolbook()
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
