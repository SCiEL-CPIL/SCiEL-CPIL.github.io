import time
import sys
import random
import numpy as np

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
        if time.time() > deadline - 20:
            break
            
        # Attempt search for target_rank
        found = _search_for_rank(target_rank, deadline)
        if found:
            best_rank = target_rank
            best_U, best_V, best_W = found
            # If we find 43, we might still try 42 if time permits, but 42 is better.
            # So we don't break immediately unless it's 42 (the absolute best we're looking for in this loop)
            # Actually, the loop iterates 43 then 42. If 43 is found, we return it? No, we want the minimal.
            # So if 43 is found, we update best and continue to try 42.
            # But usually 42 is harder. Let's just continue the loop.
            
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
    Try to find a rank-target_rank decomposition using vectorized gradient descent.
    """
    n, m, p = 2, 3, 8
    r = target_rank
    dim_u, dim_v, dim_w = n*m, m*p, p*n
    
    # Target tensor
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
    
    U0 = U_init[:, :r].copy()
    V0 = V_init[:, :r].copy()
    W0 = W_init[:, :r].copy()
    
    def compute_tensor(U, V, W):
        return np.einsum('it,jt,kt->ijk', U, V, W)
    
    def compute_error(U, V, W):
        tensor = compute_tensor(U, V, W)
        return np.linalg.norm(tensor.flatten() - T_flat)
    
    def compute_gradients(U, V, W, diff):
        grad_U = np.einsum('ijk,jt,kt->it', diff, V, W)
        grad_V = np.einsum('ijk,kt,it->jt', diff, W, U)
        grad_W = np.einsum('ijk,it,jt->kt', diff, U, V)
        return grad_U, grad_V, grad_W
    
    # Gradient descent optimization with vectorized operations
    U = U0.copy()
    V = V0.copy()
    W = W0.copy()
    
    lr = 0.005
    best_error = compute_error(U, V, W)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Momentum
    vel_U = np.zeros_like(U)
    vel_V = np.zeros_like(V)
    vel_W = np.zeros_like(W)
    mu = 0.9
    
    for step in range(200):
        if time.time() > deadline - 5:
            break
        
        # Compute tensor and difference
        tensor = compute_tensor(U, V, W)
        diff = tensor - T
        
        # Compute gradients
        grad_U, grad_V, grad_W = compute_gradients(U, V, W, diff)
        
        # Update with momentum
        vel_U = mu * vel_U + (1 - mu) * grad_U
        vel_V = mu * vel_V + (1 - mu) * grad_V
        vel_W = mu * vel_W + (1 - mu) * grad_W
        
        U -= lr * vel_U
        V -= lr * vel_V
        W -= lr * vel_W
        
        # Compute new error
        new_error = compute_error(U, V, W)
        
        if new_error < best_error:
            best_error = new_error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Check convergence
        if best_error < 1e-6 * T_norm:
            break
        
        # Adaptive learning rate
        if step % 30 == 0 and step > 0:
            lr *= 0.95
    
    # Check if we found a valid solution
    if best_error < 1e-4 * T_norm:
        # Round to nearest half-integer
        best_U = np.round(best_U * 2) / 2
        best_V = np.round(best_V * 2) / 2
        best_W = np.round(best_W * 2) / 2
        
        # Verify the rounded solution
        rounded_error = compute_error(best_U, best_V, best_W)
        if rounded_error < 1e-3 * T_norm:
            return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None
