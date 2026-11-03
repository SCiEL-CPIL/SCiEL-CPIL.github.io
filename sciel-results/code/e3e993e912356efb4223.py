"""Rank-minimal tensor decomposition search for 3x3x7 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _target_tensor(n, m, p):
    """Build the target matmul structure tensor."""
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0
    return target


def _tensor_residual(U, V, W, target=None):
    """Compute target - sum_c U[:,c] V[:,c] W[:,c]."""
    n, m, p = DIMENSIONS
    if target is None:
        target = _target_tensor(n, m, p)
    result = np.einsum('ar,br,cr->abc', U, V, W)
    return target - result


def _residual_norm(U, V, W, target=None):
    R = _tensor_residual(U, V, W, target)
    return np.linalg.norm(R)


def _als_rank1_approx(target, n, m, p, init=None, n_iter=50):
    """ALS for rank-1 tensor approximation of target."""
    if init is not None:
        u, v, w = init
    else:
        u = np.random.randn(n * m)
        v = np.random.randn(m * p)
        w = np.random.randn(p * n)
    
    for _ in range(n_iter):
        # Update u
        denom = np.sum(v**2) * np.sum(w**2)
        if denom > 1e-15:
            u_new = np.zeros(n * m)
            for a in range(n * m):
                u_new[a] = np.sum(target[a, :, :] * np.outer(v, w)) / denom
            u = u_new
        
        # Update v
        denom = np.sum(u**2) * np.sum(w**2)
        if denom > 1e-15:
            v_new = np.zeros(m * p)
            for b in range(m * p):
                v_new[b] = np.sum(target[:, b, :] * np.outer(u, w)) / denom
            v = v_new
        
        # Update w
        denom = np.sum(u**2) * np.sum(v**2)
        if denom > 1e-15:
            w_new = np.zeros(p * n)
            for c in range(p * n):
                w_new[c] = np.sum(target[:, :, c] * np.outer(u, v)) / denom
            w = w_new
    
    return u, v, w


def _greedy_column_elimination(U, V, W, target, tol=1e-10, deadline=None):
    """Remove redundant columns."""
    r = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for _ in range(200):
        if deadline is not None and time.time() > deadline:
            break
        removed = False
        for c in range(r):
            if deadline is not None and time.time() > deadline:
                break
            U_t = U.copy()
            V_t = V.copy()
            W_t = W.copy()
            U_t[:, c] = 0
            V_t[:, c] = 0
            W_t[:, c] = 0
            
            R = _tensor_residual(U_t, V_t, W_t, target)
            if np.linalg.norm(R) < tol:
                U = np.delete(U, c, axis=1)
                V = np.delete(V, c, axis=1)
                W = np.delete(W, c, axis=1)
                r = U.shape[1]
                removed = True
                break
        if not removed:
            break
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Build target tensor
    target = _target_tensor(n, m, p)
    
    # Start with schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Verify schoolbook
    R = _tensor_residual(U, V, W, target)
    assert np.linalg.norm(R) < 1e-10, f"Schoolbook residual: {np.linalg.norm(R)}"
    
    # Attempt 1: Greedy column elimination
    U1, V1, W1 = _greedy_column_elimination(U, V, W, target, deadline=deadline)
    r1 = U1.shape[1]
    R1 = _tensor_residual(U1, V1, W1, target)
    if np.linalg.norm(R1) < 1e-10 and r1 < best_rank:
        best_rank = r1
        best_U, best_V, best_W = U1.copy(), V1.copy(), W1.copy()
    
    # Attempt 2: Random restart with ALS-based optimization
    if time.time() < deadline - 30:
        np.random.seed(42)
        random.seed(42)
        
        for restart in range(10):
            if time.time() > deadline - 15:
                break
            
            # Try to find a decomposition of rank r_try
            for r_try in range(15, best_rank):
                if time.time() > deadline - 10:
                    break
                
                U_r = np.random.randn(n * m, r_try)
                V_r = np.random.randn(m * p, r_try)
                W_r = np.random.randn(p * n, r_try)
                
                # Coordinate descent: update each column
                for it in range(30):
                    if time.time() > deadline - 10:
                        break
                    for c in range(r_try):
                        if time.time() > deadline - 10:
                            break
                        
                        # Residual without column c
                        U_no = U_r.copy()
                        V_no = V_r.copy()
                        W_no = W_r.copy()
                        U_no[:, c] = 0
                        V_no[:, c] = 0
                        W_no[:, c] = 0
                        
                        R_no = _tensor_residual(U_no, V_no, W_no, target)
                        
                        # ALS to find best rank-1 approximation
                        u, v, w = _als_rank1_approx(
                            R_no, n, m, p,
                            init=(U_r[:, c], V_r[:, c], W_r[:, c]),
                            n_iter=30
                        )
                        
                        U_r[:, c] = u
                        V_r[:, c] = v
                        W_r[:, c] = w
                
                # Check final residual
                R_final = _tensor_residual(U_r, V_r, W_r, target)
                err = np.linalg.norm(R_final)
                if err < 1e-8:
                    # Valid decomposition found
                    # Try to eliminate columns
                    U_r, V_r, W_r = _greedy_column_elimination(U_r, V_r, W_r, target, deadline=deadline)
                    r_final = U_r.shape[1]
                    R_final = _tensor_residual(U_r, V_r, W_r, target)
                    if np.linalg.norm(R_final) < 1e-10 and r_final < best_rank:
                        best_rank = r_final
                        best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                        break  # Try lower rank
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
