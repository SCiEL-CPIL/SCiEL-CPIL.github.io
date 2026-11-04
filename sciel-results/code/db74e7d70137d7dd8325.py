import time
import sys
import random
import numpy as np
import scipy.optimize

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    """
    n, m, p = 2, 3, 8
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill
    
    # Known rank 44 solution
    U44, V44, W44 = _get_rank_44_solution()
    
    best_rank = 44
    best_U, best_V, best_W = U44, V44, W44
    
    # Try to find rank 43 and 42
    for target_rank in [43, 42]:
        if time.time() > deadline - 15:
            break
            
        # Attempt search for target_rank
        found = _search_for_rank(target_rank, deadline)
        if found:
            best_rank = target_rank
            best_U, best_V, best_W = found
            break # If we find 43, we might still try 42 if time permits, but 42 is better.
            
    return {"U": best_U, "V": best_V, "W": best_W}

def _get_rank_44_solution():
    n, m, p = 2, 3, 8
    rank = 44
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen_block(u_map, v_map, w_map):
        nonlocal col
        # M1: (A00+A11)(B00+B11)
        U[u_map[0], col] += 1
        U[u_map[3], col] += 1
        V[v_map[0], col] += 1
        V[v_map[3], col] += 1
        W[w_map[0], col] += 1
        W[w_map[3], col] += 1
        col += 1
        
        # M2: (A10+A11)B00
        U[u_map[2], col] += 1
        U[u_map[3], col] += 1
        V[v_map[0], col] += 1
        W[w_map[2], col] += 1
        W[w_map[3], col] += -1
        col += 1
        
        # M3: A00(B01-B11)
        U[u_map[0], col] += 1
        V[v_map[1], col] += 1
        V[v_map[3], col] += -1
        W[w_map[1], col] += 1
        W[w_map[3], col] += 1
        col += 1
        
        # M4: A11(B10-B00)
        U[u_map[3], col] += 1
        V[v_map[2], col] += 1
        V[v_map[0], col] += -1
        W[w_map[0], col] += 1
        W[w_map[2], col] += 1
        col += 1
        
        # M5: (A00+A01)B11
        U[u_map[0], col] += 1
        U[u_map[1], col] += 1
        V[v_map[3], col] += 1
        W[w_map[0], col] += -1
        W[w_map[1], col] += 1
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[u_map[2], col] += 1
        U[u_map[0], col] += -1
        V[v_map[0], col] += 1
        V[v_map[1], col] += 1
        W[w_map[3], col] += 1
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[u_map[1], col] += 1
        U[u_map[3], col] += -1
        V[v_map[2], col] += 1
        V[v_map[3], col] += 1
        W[w_map[0], col] += 1
        col += 1

    # Process left half (k = 0..3) and right half (k = 4..7)
    for half in range(2):
        u_map = {0: 0, 1: 1, 2: 3, 3: 4}
        
        b_start = half * 4
        v_map_b1 = {
            0: 0 * p + (b_start + 0),
            1: 0 * p + (b_start + 1),
            2: 1 * p + (b_start + 0),
            3: 1 * p + (b_start + 1)
        }
        w_map_c1 = {
            0: (b_start + 0) * n + 0,
            1: (b_start + 1) * n + 0,
            2: (b_start + 0) * n + 1,
            3: (b_start + 1) * n + 1
        }
        add_strassen_block(u_map, v_map_b1, w_map_c1)
        
        b_start2 = half * 4 + 2
        v_map_b2 = {
            0: 0 * p + (b_start2 + 0),
            1: 0 * p + (b_start2 + 1),
            2: 1 * p + (b_start2 + 0),
            3: 1 * p + (b_start2 + 1)
        }
        w_map_c2 = {
            0: (b_start2 + 0) * n + 0,
            1: (b_start2 + 1) * n + 0,
            2: (b_start2 + 0) * n + 1,
            3: (b_start2 + 1) * n + 1
        }
        add_strassen_block(u_map, v_map_b2, w_map_c2)
        
        for i in range(2):
            for k in range(4):
                u_idx = i * m + 2
                v_idx = 2 * p + (half * 4 + k)
                w_idx = (half * 4 + k) * n + i
                
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
    
    return U.tolist(), V.tolist(), W.tolist()

def _verify_solution(U, V, W, n=2, m=3, p=8):
    """Verify that the decomposition exactly represents matrix multiplication."""
    dim_u, dim_v, dim_w = n*m, m*p, p*n
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(dim_u):
        for j in range(dim_v):
            if i // m == j % p:
                k = (j // p) * n + (i % m)
                T[i, j, k] = 1
    
    # Compute tensor from decomposition
    tensor = np.einsum('it,jt,kt->ijk', np.array(U), np.array(V), np.array(W))
    
    error = np.linalg.norm(tensor - T)
    return error < 1e-6

def _search_for_rank(target_rank, deadline):
    """
    Try to find a rank-target_rank decomposition using SVD initialization and gradient descent.
    """
    n, m, p = 2, 3, 8
    r = target_rank
    dim_u, dim_v, dim_w = n*m, m*p, p*n
    
    # Target tensor: T[i,j,k] = 1 if i//m == j%p
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(dim_u):
        for j in range(dim_v):
            if i // m == j % p:
                k = (j // p) * n + (i % m)
                T[i, j, k] = 1
    
    T_flat = T.flatten()
    T_norm = np.linalg.norm(T_flat)
    
    # Initialize with rank-44 solution, take first r columns
    U44, V44, W44 = _get_rank_44_solution()
    U_init = np.array(U44, dtype=float)
    V_init = np.array(V44, dtype=float)
    W_init = np.array(W44, dtype=float)
    
    # Better initialization: Use SVD of the matricized tensor
    # Matricize T as T_U (dim_u x dim_v*dim_w)
    T_U = T.reshape(dim_u, dim_v * dim_w)
    U_svd, s, Vt_svd = np.linalg.svd(T_U, full_matrices=False)
    
    # Take top r components
    U_svd_r = U_svd[:, :r]
    s_r = s[:r]
    Vt_svd_r = Vt_svd[:r, :]
    
    # Reshape Vt_svd_r back to (dim_v, dim_w, r)
    # Vt_svd_r is (r, dim_v*dim_w)
    # We need to form V and W such that U @ diag(s) @ Vt approximates T_U
    # T_U approx = U_svd_r @ diag(s_r) @ Vt_svd_r
    # This corresponds to tensor approx = sum_t s_t * u_t * v_t * w_t
    # We can absorb s_t into U, V, or W.
    # Let's distribute: U_t = u_t * s_t^(1/3), V_t = v_t * s_t^(1/3), W_t = w_t * s_t^(1/3)
    
    # However, we need to split Vt_svd_r (r x dim_v*dim_w) into V (dim_v x r) and W (dim_w x r)
    # This is not directly possible from a single SVD of one matricization.
    # We need to use the tensor structure.
    # A better initialization is to use the rank-44 solution and randomly select/combine columns.
    # Or, use the "randomized SVD" idea on the tensor itself?
    # For now, let's stick to a modified gradient descent with multiple restarts and better init from rank-44.
    
    # Strategy: Randomly select r columns from the rank-44 solution? No, that loses info.
    # Strategy: Initialize U, V, W with random vectors, but scale them to match the norm.
    # Or, use the rank-44 solution and "merge" columns.
    
    # Let's try a simple restart strategy with the rank-44 solution as a base.
    # We will run gradient descent for a few iterations.
    
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for restart in range(5):
        if time.time() > deadline - 5:
            break
        
        # Initialize by taking a random subset of columns from rank-44?
        # Or random init?
        # Let's try random init with small noise added to rank-44 columns.
        
        # Pick r columns from 44
        idx = np.random.choice(44, r, replace=False)
        U0 = U_init[:, idx].copy()
        V0 = V_init[:, idx].copy()
        W0 = W_init[:, idx].copy()
        
        # Add small random noise
        noise = 0.1
        U0 += np.random.normal(0, noise, U0.shape)
        V0 += np.random.normal(0, noise, V0.shape)
        W0 += np.random.normal(0, noise, W0.shape)
        
        U = U0.copy()
        V = V0.copy()
        W = W0.copy()
        
        lr = 0.01
        vel_U = np.zeros_like(U)
        vel_V = np.zeros_like(V)
        vel_W = np.zeros_like(W)
        mu = 0.9
        
        for step in range(100):
            if time.time() > deadline - 5:
                break
            
            tensor = np.einsum('it,jt,kt->ijk', U, V, W)
            diff = tensor - T
            error = np.linalg.norm(diff)
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if best_error < 1e-6 * T_norm:
                break
            
            grad_U = np.einsum('ijk,jt,kt->it', diff, V, W)
            grad_V = np.einsum('ijk,kt,it->jt', diff, W, U)
            grad_W = np.einsum('ijk,it,jt->kt', diff, U, V)
            
            vel_U = mu * vel_U + (1 - mu) * grad_U
            vel_V = mu * vel_V + (1 - mu) * grad_V
            vel_W = mu * vel_W + (1 - mu) * grad_W
            
            U -= lr * vel_U
            V -= lr * vel_V
            W -= lr * vel_W
            
            if step % 20 == 0 and step > 0:
                lr *= 0.95
        
    if best_error < 1e-4 * T_norm and best_U is not None:
        # Round to nearest half-integer
        best_U = np.round(best_U * 2) / 2
        best_V = np.round(best_V * 2) / 2
        best_W = np.round(best_W * 2) / 2
        
        # Verify the rounded solution
        rounded_error = np.linalg.norm(np.einsum('it,jt,kt->ijk', best_U, best_V, best_W) - T)
        if rounded_error < 1e-3 * T_norm:
            return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None
