"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 145.0 # Safety margin for 150s limit

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a lower rank decomposition using ALS
    # Start from rank 24 up to 27
    for rank in range(24, 28):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        success = _try_find_decomposition(n, m, p, rank)
        if success:
            return success
    
    # Fallback to known rank 28
    U, V, W = _strassen_28(n, m, p)
    return {"U": U, "V": V, "W": W}

def _try_find_decomposition(n, m, p, rank):
    """
    Attempts to find a decomposition of the given rank using Alternating Least Squares (ALS).
    Returns dict if successful, None otherwise.
    """
    # Target tensor T[i, j, k] = 1 if k == i*m + j is not right.
    # T is defined by C[i, k] = sum_j A[i, j] B[j, k]
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i (based on task description W[k*n+i, r])
    
    # We want sum_r U[u, r] V[v, r] W[w, r] = T[u, v, w]
    # Non-zeros in T:
    # For each i in 0..n-1, j in 0..m-1, k in 0..p-1:
    # If we multiply A[i, j] and B[j, k], it contributes to C[i, k].
    # So T[u, v, w] is 1 if:
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    
    # Note: The W index in the task is W[k*n + i, r].
    # So w corresponds to k*n + i.
    
    target_tensor = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_tensor[u, v, w] = 1.0

    # Initialize U, V, W with random values
    # To help convergence, we can use the Strassen solution as a starting point if rank >= 28,
    # but here rank < 28, so we use random.
    # We use half-integers or integers? The task allows half-integers.
    # ALS will produce floats. We will round at the end.
    
    np.random.seed(42)
    U = np.random.randn(n*m, rank) * 0.5
    V = np.random.randn(m*p, rank) * 0.5
    W = np.random.randn(p*n, rank) * 0.5
    
    # ALS Iterations
    max_iters = 200
    tol = 1e-5
    
    for it in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            return None
            
        # Update U
        # For each u, U[u, :] = sum_v,w T[u, v, w] V[v, :] W[w, :] / sum_v,w V[v, :] W[w, :]^T ?
        # Actually, we solve for U such that U (V^T W) approx T.
        # This is a linear least squares problem for each row of U.
        # T_u = sum_v T[u, v, :] V[v, :]^T ? No.
        
        # Let's reshape.
        # T is n*m x m*p x p*n.
        # We want U @ (V^T @ W) ? No.
        # sum_r U[u, r] V[v, r] W[w, r]
        
        # Fix V, W. Solve for U.
        # For a fixed u, we want to find U[u, :] such that:
        # sum_v,w T[u, v, w] (sum_r U[u, r] V[v, r] W[w, r])^2 is minimized? No, we want exact match.
        # We have linear equations.
        # For each (v, w) where T[u, v, w] != 0:
        # sum_r U[u, r] (V[v, r] W[w, r]) = T[u, v, w]
        
        # Let M_u be a matrix of size (num_nonzeros_u, rank) where M_u[idx, r] = V[v_idx, r] W[w_idx, r]
        # Let y_u be the vector of T[u, v_idx, w_idx]
        # Solve M_u U[u, :]^T = y_u
        
        # Precompute non-zero indices for efficiency
        nz_u = []
        for u in range(n*m):
            nz_u.append([])
            for v in range(m*p):
                for w in range(p*n):
                    if target_tensor[u, v, w] != 0:
                        nz_u[u].append((v, w, target_tensor[u, v, w]))
        
        # Update U
        for u in range(n*m):
            if not nz_u[u]: continue
            M_u = np.zeros((len(nz_u[u]), rank))
            y_u = np.zeros(len(nz_u[u]))
            for idx, (v, w, t_val) in enumerate(nz_u[u]):
                M_u[idx, :] = V[v, :] * W[w, :]
                y_u[idx] = t_val
            try:
                U[u, :], residuals, rank_u, sv = np.linalg.lstsq(M_u, y_u, rcond=None)
            except:
                pass
        
        # Update V
        nz_v = []
        for v in range(m*p):
            nz_v.append([])
            for u in range(n*m):
                for w in range(p*n):
                    if target_tensor[u, v, w] != 0:
                        nz_v[v].append((u, w, target_tensor[u, v, w]))
                        
        for v in range(m*p):
            if not nz_v[v]: continue
            M_v = np.zeros((len(nz_v[v]), rank))
            y_v = np.zeros(len(nz_v[v]))
            for idx, (u, w, t_val) in enumerate(nz_v[v]):
                M_v[idx, :] = U[u, :] * W[w, :]
                y_v[idx] = t_val
            try:
                V[v, :], residuals, rank_v, sv = np.linalg.lstsq(M_v, y_v, rcond=None)
            except:
                pass
                
        # Update W
        nz_w = []
        for w in range(p*n):
            nz_w.append([])
            for u in range(n*m):
                for v in range(m*p):
                    if target_tensor[u, v, w] != 0:
                        nz_w[w].append((u, v, target_tensor[u, v, w]))
                        
        for w in range(p*n):
            if not nz_w[w]: continue
            M_w = np.zeros((len(nz_w[w]), rank))
            y_w = np.zeros(len(nz_w[w]))
            for idx, (u, v, t_val) in enumerate(nz_w[w]):
                M_w[idx, :] = U[u, :] * V[v, :]
                y_w[idx] = t_val
            try:
                W[w, :], residuals, rank_w, sv = np.linalg.lstsq(M_w, y_w, rcond=None)
            except:
                pass
        
        # Check convergence
        # Compute error
        # T_approx = sum_r U[:, r] V[:, r] W[:, r]
        # We only check non-zeros
        error = 0.0
        for u in range(n*m):
            for v, w, t_val in nz_u[u]:
                val = np.dot(U[u, :], V[v, :] * W[w, :])
                error += (val - t_val)**2
        
        if error < tol:
            # Round to half-integers
            U_int = np.round(U * 2) / 2
            V_int = np.round(V * 2) / 2
            W_int = np.round(W * 2) / 2
            
            # Verify exactness
            valid = True
            for u in range(n*m):
                for v, w, t_val in nz_u[u]:
                    val = np.dot(U_int[u, :], V_int[v, :] * W_int[w, :])
                    if abs(val - t_val) > 1e-9:
                        valid = False
                        break
                if not valid: break
            
            if valid:
                return {
                    "U": [[float(x) for x in row] for row in U_int],
                    "V": [[float(x) for x in row] for row in V_int],
                    "W": [[float(x) for x in row] for row in W_int]
                }
    return None

def _strassen_28(n, m, p):
    """Construct the rank-28 Strassen-based decomposition."""
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = [
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(1, 1), (3, -1)]},
        {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(2, 1), (3, 1)]},
        {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(0, 1), (1, 1)]},
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (2, 1)]},
        {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
        {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]}
    ]
    
    blocks = [
        {'j_offset': 0, 'k_offset': 0},
        {'j_offset': 2, 'k_offset': 0},
        {'j_offset': 0, 'k_offset': 2},
        {'j_offset': 2, 'k_offset': 2}
    ]
    
    col = 0
    for block in blocks:
        j_off = block['j_offset']
        k_off = block['k_offset']
        for term in strassen_terms:
            for (idx, coeff) in term['U']:
                i = idx // 2
                j = idx % 2
                global_i = i
                global_j = j + j_off
                U[global_i * m + global_j][col] += coeff
            for (idx, coeff) in term['V']:
                j = idx // 2
                k = idx % 2
                global_j = j + j_off
                global_k = k + k_off
                V[global_j * p + global_k][col] += coeff
            for (idx, coeff) in term['W']:
                k = idx // 2
                i = idx % 2
                global_k = k + k_off
                global_i = i
                W[global_k * n + global_i][col] += coeff
            col += 1
    
    U = [[int(x) for x in row] for row in U]
    V = [[int(x) for x in row] for row in V]
    W = [[int(x) for x in row] for row in W]
    
    return U, V, W
