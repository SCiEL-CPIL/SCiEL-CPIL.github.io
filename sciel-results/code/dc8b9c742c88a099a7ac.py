"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s kill
    
    # Try to build a Strassen-based decomposition
    # For 2x2 x 2xp multiplication, use the 7 Strassen products
    # Each Strassen product involves linear combinations of A and B entries
    # A is 2x2, B is 2xp
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # Strassen 7 products for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    #
    # In our case, B00, B01, B10, B11 are 1xp vectors
    # So each Mi is 1xp, and Ci is 1xp
    # Total rank = 7 * p = 77 (worse than schoolbook 44)
    #
    # Need a better approach. Let me try direct optimization.
    
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # Start with schoolbook
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
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank by finding linear dependencies
    # For each candidate rank from 39 down to 30, try to find a valid decomposition
    for target_rank in range(39, max(29, 39 - 10), -1):
        if time.time() > deadline - 30:
            break
        result = _try_rank(n, m, p, target_rank, deadline - 20)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break
    
    return best_result


def _try_rank(n, m, p, target_rank, time_limit):
    """Try to find a valid decomposition of the given rank."""
    deadline = time.time() + time_limit
    
    # Initialize with random matrices
    for attempt in range(5):
        if time.time() > deadline:
            break
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        # Try to optimize using simple gradient descent or use a structured initialization
        # For now, just verify if this random guess happens to work (unlikely)
        if _verify(U, V, W, n, m, p, tol=1e-8):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None


def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that the tensor decomposition is correct."""
    try:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        if U_arr.ndim == 1:
            U_arr = U_arr.reshape(-1, 1)
        if V_arr.ndim == 1:
            V_arr = V_arr.reshape(-1, 1)
        if W_arr.ndim == 1:
            W_arr = W_arr.reshape(-1, 1)
        rank = U_arr.shape[1]
        U_r = U_arr.reshape(n, m, rank)
        V_r = V_arr.reshape(m, p, rank)
        W_r = W_arr.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T, 1.0, atol=tol)
    except Exception:
        return False
# EVOLVE-BLOCK-END
