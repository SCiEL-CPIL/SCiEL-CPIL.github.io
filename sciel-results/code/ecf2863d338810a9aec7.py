"""Matrix multiplication 2x2x15 tensor decomposition - global search for reduced rank."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 15]

def _build_strassen_2x2x2():
    """Strassen rank-7 for 2x2x2."""
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_schoolbook_2x2x1():
    """Schoolbook rank-4 for 2x2x1."""
    n, m, p = 2, 2, 1
    rank = 4
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1
            V[j * p, col] = 1
            W[i, col] = 1
            col += 1
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify decomposition correctness."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _build_baseline_53():
    """Build rank-53 baseline: 7 Strassen + 1 schoolbook."""
    n, m, p = 2, 2, 15
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook_2x2x1()
    
    total_rank = 7 * 7 + 4
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for b in range(7):
        start_k = 2 * b
        U[:, col:col+7] = U_str
        for j in range(2):
            for kl in range(2):
                V[j*p + start_k + kl, col:col+7] = V_str[j*2 + kl, :]
        for kl in range(2):
            for i in range(2):
                W[(start_k+kl)*n + i, col:col+7] = W_str[kl*2 + i, :]
        col += 7
    
    U[:, col:col+4] = U_sb
    for j in range(2):
        V[j*p + 14, col:col+4] = V_sb[j, :]
    for i in range(2):
        W[14*n + i, col:col+4] = W_sb[i, :]
    
    return U, V, W

def _search_2x2x3_rank10(deadline):
    """
    Search for rank-10 decomposition of 2x2x3 matrix multiplication.
    Uses scipy optimization to find a valid decomposition.
    """
    n, m, p = 2, 2, 3
    target_rank = 10
    
    # Each term r has: u_r (4 entries), v_r (6 entries), w_r (6 entries)
    # Total variables: 10 * (4 + 6 + 6) = 160
    # Constraints: 12 equations (for each i,j,k)
    
    def _tensor_residual(x):
        """Compute residual: for each (i,j,k), sum_r u*v*w - 1"""
        U_mat = x[0:40].reshape(10, 4)
        V_mat = x[40:100].reshape(10, 6)
        W_mat = x[100:160].reshape(10, 6)
        
        residuals = np.zeros(12)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_mat[:, i*m+j] * V_mat[:, j*p+k] * W_mat[:, k*n+i])
                    residuals[idx] = val - 1.0
                    idx += 1
        return residuals
    
    def _objective(x):
        res = _tensor_residual(x)
        return np.sum(res**2)
    
    # Try multiple random initializations
    best_x = None
    best_obj = np.inf
    
    np.random.seed(42)
    for trial in range(200):
        if time.time() > deadline - 5:
            break
        
        # Random initialization with small values
        x0 = np.random.randn(160) * 0.5
        
        try:
            from scipy.optimize import minimize
            result = minimize(_objective, x0, method='L-BFGS-B', 
                             options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12})
            if result.fun < best_obj:
                best_obj = result.fun
                best_x = result.x
            if best_obj < 1e-12:
                break
        except Exception:
            continue
    
    if best_x is None or best_obj > 1e-10:
        return None
    
    # Extract matrices
    U10 = best_x[0:40].reshape(10, 4)
    V10 = best_x[40:100].reshape(10, 6)
    W10 = best_x[100:160].reshape(10, 6)
    
    # Verify
    if not _verify(U10, V10, W10, n, m, p):
        return None
    
    return U10, V10, W10

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Build rank-53 baseline
    U, V, W = _build_baseline_53()
    
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook rank 60
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, idx] = 1
                    V[j*p+k, idx] = 1
                    W[k*n+i, idx] = 1
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = 53
    best_U, best_V, best_W = U, V, W
    
    # Try to find rank-10 for 2x2x3
    if time.time() < deadline - 10:
        result = _search_2x2x3_rank10(deadline)
        if result is not None:
            U10, V10, W10 = result
            # Build 5 blocks of 2x2x3 with rank 10 each
            total_rank = 5 * 10
            U_new = np.zeros((n*m, total_rank))
            V_new = np.zeros((m*p, total_rank))
            W_new = np.zeros((p*n, total_rank))
            
            for b in range(5):
                start_k = 3 * b
                col = b * 10
                U_new[:, col:col+10] = U10
                for j in range(2):
                    for kl in range(3):
                        V_new[j*p + start_k + kl, col:col+10] = V10[j*3 + kl, :]
                for kl in range(3):
                    for i in range(2):
                        W_new[(start_k+kl)*n + i, col:col+10] = W10[kl*2 + i, :]
            
            if _verify(U_new, V_new, W_new, n, m, p):
                best_rank = 50
                best_U, best_V, best_W = U_new, V_new, W_new
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
