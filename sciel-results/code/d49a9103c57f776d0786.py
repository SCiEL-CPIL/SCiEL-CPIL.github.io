"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # 5 second safety margin
    
    # Try ranks from 44 upward (44 is known achievable)
    for rank in range(44, 49):
        if time.time() > deadline:
            break
        for seed in range(5):
            if time.time() > deadline:
                break
            try:
                result = _find_decomposition(n, m, p, rank, seed)
                if result is not None:
                    return result
            except Exception:
                continue
    
    # Fallback to trivial solution
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _find_decomposition(n, m, p, rank, seed):
    """Find rank-r decomposition using alternating least squares."""
    np.random.seed(seed)
    
    U = np.random.randn(n * m, rank) * 0.5
    V = np.random.randn(m * p, rank) * 0.5
    W = np.random.randn(p * n, rank) * 0.5
    
    start = time.time()
    max_time = 8  # seconds per attempt
    
    for iteration in range(300):
        if time.time() - start > max_time:
            break
        
        # Solve for U: fix (i,j), vary (jp, k, ip, kp)
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                num_eq = m * p * n * p
                A = np.zeros((num_eq, rank))
                b = np.zeros(num_eq)
                idx = 0
                for jp in range(m):
                    for k in range(p):
                        for ip in range(n):
                            for kp in range(p):
                                A[idx, :] = V[jp * p + k, :] * W[kp * n + ip, :]
                                b[idx] = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                idx += 1
                U_new[i * m + j, :], _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
        U = U_new
        
        # Solve for V: fix (jp, k), vary (i, j, ip, kp)
        V_new = np.zeros_like(V)
        for jp in range(m):
            for k in range(p):
                num_eq = n * m * n * p
                A = np.zeros((num_eq, rank))
                b = np.zeros(num_eq)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for ip in range(n):
                            for kp in range(p):
                                A[idx, :] = U[i * m + j, :] * W[kp * n + ip, :]
                                b[idx] = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                idx += 1
                V_new[jp * p + k, :], _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
        V = V_new
        
        # Solve for W: fix (kp, ip), vary (i, j, jp, k)
        W_new = np.zeros_like(W)
        for kp in range(p):
            for ip in range(n):
                num_eq = n * m * m * p
                A = np.zeros((num_eq, rank))
                b = np.zeros(num_eq)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for jp in range(m):
                            for k in range(p):
                                A[idx, :] = U[i * m + j, :] * V[jp * p + k, :]
                                b[idx] = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                idx += 1
                W_new[kp * n + ip, :], _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
        W = W_new
        
        # Check convergence every 50 iterations
        if iteration % 50 == 49:
            err = _compute_error(U, V, W, n, m, p)
            if err < 1e-10:
                return _make_result(U, V, W)
    
    # Final check
    err = _compute_error(U, V, W, n, m, p)
    if err < 1e-10:
        return _make_result(U, V, W)
    return None

def _compute_error(U, V, W, n, m, p):
    """Compute max absolute error in the tensor decomposition."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            val = np.dot(U[i*m+j, :], V[jp*p+k, :] * W[kp*n+ip, :])
                            expected = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                            err = abs(val - expected)
                            if err > max_err:
                                max_err = err
    return max_err

def _make_result(U, V, W):
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
