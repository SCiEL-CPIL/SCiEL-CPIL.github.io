"""Program that searches for a lower-rank exact tensor decomposition for 2x3 * 3x9."""
import random
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def _construct_tensor(n, m, p):
    """Return the matrix multiplication tensor as a 3-index array T[i, j, k]."""
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _build_decomposition(n, m, p, rank, U, V, W):
    """
    Build factor matrices from parameter arrays.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    return U, V, W

def _decomposition_error(n, m, p, U, V, W):
    """
    Compute the Frobenius norm of the difference between the tensor
    and its rank-decomposition approximation.
    
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    
    # Reshape for efficient computation
    # U: (n, m, rank)
    # V: (m, p, rank)
    # W: (p, n, rank)
    U_r = U.reshape(n, m, -1)
    V_r = V.reshape(m, p, -1)
    W_r = W.reshape(p, n, -1)
    
    # Compute approximated tensor
    # T_approx[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    T_approx = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    
    # True tensor
    T_true = np.ones((n, m, p), dtype=np.float64)
    
    # Error
    diff = T_true - T_approx
    return np.linalg.norm(diff)

def _try_random_rank(n, m, p, rank, max_iters, deadline, rng):
    """
    Try to find a valid rank decomposition using random initialization + gradient descent.
    Returns (U, V, W) if found, else None.
    """
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    U = rng.standard_normal((n * m, rank))
    V = rng.standard_normal((m * p, rank))
    W = rng.standard_normal((p * n, rank))
    
    T_true = np.ones((n, m, p), dtype=np.float64)
    
    lr = 0.01
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        
        T_approx = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        diff = T_true - T_approx
        err = np.linalg.norm(diff)
        
        if err < best_err:
            best_err = err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if err < 1e-9:
            return U.copy(), V.copy(), W.copy()
        
        # Gradients
        dU = np.einsum('ijk,jkc,kic->ijc', diff, V_r, W_r)
        dV = np.einsum('ijk,ijc,kic->jkc', diff, U_r, W_r)
        dW = np.einsum('ijk,ijc,jkc->kic', diff, U_r, V_r)
        
        U -= lr * dU.reshape(n * m, rank)
        V -= lr * dV.reshape(m * p, rank)
        W -= lr * dW.reshape(p * n, rank)
    
    if best_err < 1e-9:
        return best_U, best_V, best_W
    return None

def _check_decomposition(n, m, p, U, V, W, tol=1e-9):
    """Verify that the decomposition is exact."""
    U_r = np.asarray(U, dtype=np.float64).reshape(n, m, -1)
    V_r = np.asarray(V, dtype=np.float64).reshape(m, p, -1)
    W_r = np.asarray(W, dtype=np.float64).reshape(p, n, -1)
    
    T_approx = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    T_true = np.ones((n, m, p), dtype=np.float64)
    
    diff = T_true - T_approx
    return np.max(np.abs(diff)) < tol

def _snap_to_halfint(arr, tol=1e-6):
    """Snap values close to half-integers to exact half-integers."""
    arr = np.asarray(arr, dtype=np.float64)
    snapped = np.round(arr * 2) / 2.0
    if np.max(np.abs(arr - snapped)) < tol:
        return snapped
    return arr

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Deadline guard
    deadline = time.time() + 120.0
    
    rng = random.Random(42)
    np_rng = np.random.RandomState(42)
    
    # Try decreasing ranks
    # Start from some heuristic lower bound
    # Lander-Motzkin lower bound: n*m + m*p + p*n - 1
    lb = n * m + m * p + p * n - 1
    
    # Try ranks from schoolbook down to lb+1
    # But only try a few promising ranks
    ranks_to_try = []
    
    # First, try to find if there's a known better rank
    # For 2x3 * 3x9, the rank is at most 54 (schoolbook)
    # Try some candidate ranks
    for r in range(schoolbook_rank - 1, max(schoolbook_rank - 20, lb), -1):
        ranks_to_try.append(r)
    
    # Add the lower bound itself
    if lb not in ranks_to_try:
        ranks_to_try.append(lb)
    
    for rank in ranks_to_try:
        if time.time() > deadline - 5.0:
            break
        
        # Try random restarts for this rank
        max_restarts = 5 if rank >= 50 else 10
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
            
            max_iters = 2000
            result = _try_random_rank(n, m, p, rank, max_iters, deadline - 5.0, np_rng)
            
            if result is not None:
                U, V, W = result
                # Snap to half-integers
                U_s = _snap_to_halfint(U)
                V_s = _snap_to_halfint(V)
                W_s = _snap_to_halfint(W)
                
                if _check_decomposition(n, m, p, U_s, V_s, W_s, tol=1e-6):
                    return {
                        "U": U_s.tolist(),
                        "V": V_s.tolist(),
                        "W": W_s.tolist()
                    }
    
    # Fall back to schoolbook
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
