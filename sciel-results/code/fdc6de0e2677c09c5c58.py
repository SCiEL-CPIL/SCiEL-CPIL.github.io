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
    deadline = start_time + 110.0  # 10s margin
    
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

def _search_for_rank(target_rank, deadline):
    """
    Try to find a rank-target_rank decomposition using border rank relaxation.
    
    Instead of trying to eliminate terms from the rank-44 solution, we directly
    optimize for a rank-r decomposition by minimizing the distance to the rank-r
    secant variety. This can find solutions that aren't simple linear combinations
    of the existing terms.
    """
    n, m, p = 2, 3, 8
    r = target_rank
    
    # Target tensor: T[i,j,k] = 1 if i//m == j%p
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n*m):
        for j in range(m*p):
            if i // m == j % p:
                k = (j // p) * n + (i % m)
                T[i, j, k] = 1
    
    # Flatten T for optimization
    T_flat = T.flatten()
    T_norm = np.linalg.norm(T_flat)
    
    # Initialize with random matrices (or use rank-44 as starting point)
    U44, V44, W44 = _get_rank_44_solution()
    U_init = np.array(U44, dtype=float)
    V_init = np.array(V44, dtype=float)
    W_init = np.array(W44, dtype=float)
    
    # Use first r columns as initialization
    U0 = U_init[:, :r].copy()
    V0 = V_init[:, :r].copy()
    W0 = W_init[:, :r].copy()
    
    def compute_tensor(U, V, W):
        """Compute the tensor from U, V, W matrices."""
        tensor = np.zeros((n*m, m*p, p*n))
        for t in range(r):
            tensor += np.einsum('i,j,k->ijk', U[:, t], V[:, t], W[:, t])
        return tensor
    
    def compute_error(U, V, W):
        """Compute the error between computed tensor and target."""
        tensor = compute_tensor(U, V, W)
        return np.linalg.norm(tensor.flatten() - T_flat)
    
    # Gradient descent optimization
    U = U0.copy()
    V = V0.copy()
    W = W0.copy()
    
    lr = 0.01
    best_error = compute_error(U, V, W)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for step in range(500):
        if time.time() > deadline - 5:
            break
        
        # Compute gradient
        tensor = compute_tensor(U, V, W)
        diff = tensor.flatten() - T_flat
        
        # Gradient with respect to U
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for t in range(r):
            # d/dU[i,t] sum_j,k U[i,t]*V[j,t]*W[k,t] - T[i,j,k]
            # = V[j,t]*W[k,t] for each (i,j,k)
            for i in range(n*m):
                for j in range(m*p):
                    for k in range(p*n):
                        grad_U[i, t] += diff[i*m*p + j*p*n + k] * V[j, t] * W[k, t]
                        grad_V[j, t] += diff[i*m*p + j*p*n + k] * U[i, t] * W[k, t]
                        grad_W[k, t] += diff[i*m*p + j*p*n + k] * U[i, t] * V[j, t]
        
        # Update with momentum-like decay
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Compute new error
        new_error = compute_error(U, V, W)
        
        if new_error < best_error:
            best_error = new_error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Check convergence
        if best_error < 1e-8 * T_norm:
            break
        
        # Adaptive learning rate
        if step % 50 == 0 and step > 0:
            lr *= 0.9
    
    # Check if we found a valid solution
    if best_error < 1e-6 * T_norm:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None
