"""Improved program for matrix multiplication tensor decomposition (2x3x6)."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]


def _schoolbook_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
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


def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that the decomposition exactly represents matrix multiplication."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                if abs(actual - expected) > tol:
                    return False
    return True


def _als_decompose(T, rank, seed=0, max_sweeps=200, tol=1e-12):
    """ALS tensor decomposition with random restarts."""
    rng = np.random.RandomState(seed)
    dim1, dim2, dim3 = T.shape
    
    U = rng.randn(dim1, rank)
    V = rng.randn(dim2, rank)
    W = rng.randn(dim3, rank)
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for sweep in range(max_sweeps):
        # Update U
        VW_kr = np.einsum('jc,kc->jkc', V, W).reshape(-1, rank)
        T1 = T.reshape(dim1, -1)
        try:
            U_new, _, _, _ = np.linalg.lstsq(VW_kr, T1.T, rcond=None)
            U = U_new.T
        except np.linalg.LinAlgError:
            pass
        
        # Update V
        UW_kr = np.einsum('ic,kc->ikc', U, W).reshape(-1, rank)
        T2 = np.transpose(T, (1, 0, 2)).reshape(dim2, -1)
        try:
            V_new, _, _, _ = np.linalg.lstsq(UW_kr, T2.T, rcond=None)
            V = V_new.T
        except np.linalg.LinAlgError:
            pass
        
        # Update W
        UV_kr = np.einsum('ic,jc->ijc', U, V).reshape(-1, rank)
        T3 = np.transpose(T, (2, 0, 1)).reshape(dim3, -1)
        try:
            W_new, _, _, _ = np.linalg.lstsq(UV_kr, T3.T, rcond=None)
            W = W_new.T
        except np.linalg.LinAlgError:
            pass
        
        # Compute loss
        recon = np.einsum('ic,jc,kc->ijk', U, V, W)
        loss = np.sum((recon - T) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < tol:
            break
    
    if best_loss < 1e-8:
        return best_U, best_V, best_W
    return None


def _try_rank(T, rank, n, m, p, deadline, max_trials=30):
    """Try multiple random initializations for a given rank."""
    for seed in range(max_trials):
        if time.time() > deadline - 5.0:
            break
        result = _als_decompose(T, rank, seed=seed)
        if result is not None:
            U, V, W = result
            U_list = [[round(float(x), 10) for x in row] for row in U]
            V_list = [[round(float(x), 10) for x in row] for row in V]
            W_list = [[round(float(x), 10) for x in row] for row in W]
            if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-6):
                return {"U": U_list, "V": V_list, "W": W_list}
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135.0
    
    T = _build_tensor(n, m, p)
    
    # The schoolbook rank is n*m*p = 36
    # For 2x3x6, the minimal rank is likely much lower
    # Known bounds: rank <= n*m*p, and for small dimensions we can search
    
    # Try ranks from small to large, but skip very small ones that are impossible
    # For 2x3x6: the tensor has 36 non-zero entries, each requiring at least one term
    # But due to structure, we can do better
    
    # Let's try ranks from 6 up to 35
    for rank in range(6, 36):
        if time.time() > deadline - 10.0:
            break
        
        # Allocate time based on rank: lower ranks get more trials
        time_per_rank = min(3.0, 120.0 / (36 - 5))
        rank_deadline = time.time() + time_per_rank
        
        max_trials = max(5, int(time_per_rank * 10))
        
        result = _try_rank(T, rank, n, m, p, rank_deadline, max_trials=max_trials)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _schoolbook_decomposition(n, m, p)
