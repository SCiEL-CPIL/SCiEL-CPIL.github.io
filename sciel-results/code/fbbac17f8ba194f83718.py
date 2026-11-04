import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    start_time = time.time()
    deadline = start_time + 110.0

    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb

    # Try ranks from 37 upwards. 
    # If we find a lower rank, we return it.
    # We prioritize lower ranks.
    target_ranks = [37, 38, 39, 40, 41, 42, 43, 44]
    
    for target_rank in target_ranks:
        if target_rank >= best_rank:
            continue
        if time.time() > deadline - 20:
            break
            
        found = False
        # More restarts for lower ranks as they are harder
        max_restarts = 10 if target_rank <= 38 else 5
        
        for restart in range(max_restarts):
            if time.time() > deadline - 20:
                break
            try:
                result = search_rank_vectorized(n, m, p, target_rank, deadline)
                if result is not None:
                    return result
            except Exception:
                pass

    return best_result

def search_rank_vectorized(n, m, p, rank, deadline):
    """
    Uses vectorized Gradient Descent to minimize the Frobenius norm of the tensor decomposition error.
    """
    # Random initialization with small values
    # Using half-integers or small integers helps snapping later
    U = np.random.randint(-2, 3, size=(n * m, rank)) * 0.5
    V = np.random.randint(-2, 3, size=(m * p, rank)) * 0.5
    W = np.random.randint(-2, 3, size=(p * n, rank)) * 0.5
    
    # Reshape for easier indexing
    # U3: (n, m, rank)
    # V3: (m, p, rank)
    # W3: (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    max_iters = 2000
    step_size = 0.05
    
    # Precompute indices or just use einsum
    # T_ijk = sum_r U_ijk V_jkr W_kir
    # We want T_ijk = 1
    
    # Gradient of L = sum (T_ijk - 1)^2 w.r.t U_ijk
    # dL/dU_ijk = 2 * (T_ijk - 1) * V_jkr * W_kir (sum over k, r fixed? No, r is the index of U)
    # Wait, T_ijk = sum_r U_ijk[r] * V_jkr[r] * W_kir[r]
    # dT_ijk/dU_ijk[r] = V_jkr[r] * W_kir[r]
    # So dL/dU_ijk[r] = 2 * (T_ijk - 1) * V_jkr[r] * W_kir[r]
    
    # We can compute T first, then gradients.
    
    for iter in range(max_iters):
        if time.time() > deadline - 5:
            break
            
        # Compute T
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        error = np.sum((T - 1.0)**2)
        
        if error < 1e-8:
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            snap_error = compute_error(U_snap, V_snap, W_snap, n, m, p)
            if snap_error < 1e-8:
                return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            # If snap fails, continue optimizing to get closer? Or try smaller step?
            
        # Compute gradients
        # dU = 2 * (T - 1) * V * W
        # T is (n, m, p)
        # V is (m, p, rank)
        # W is (p, n, rank)
        
        # dU3[i, j, r] = 2 * sum_k (T[i, j, k] - 1) * V3[j, k, r] * W3[k, i, r]
        # dU3 = 2 * np.einsum('ijk,jkr,kir->ijr', T - 1.0, V3, W3)
        
        dU3 = 2.0 * np.einsum('ijk,jkr,kir->ijr', T - 1.0, V3, W3)
        dV3 = 2.0 * np.einsum('ijk,ijr,kir->jkr', T - 1.0, U3, W3)
        dW3 = 2.0 * np.einsum('ijk,ijr,jkr->kir', T - 1.0, U3, V3)
        
        # Update
        U3 -= step_size * dU3
        V3 -= step_size * dV3
        W3 -= step_size * dW3
        
        # Adaptive step size: if error increases, reduce step
        # We need previous error. 
        # For simplicity, let's just use a fixed small step or decay.
        if iter > 100:
            step_size *= 0.95
            
    # Final check
    error = compute_error(U, V, W, n, m, p)
    if error < 1e-8:
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        snap_error = compute_error(U_snap, V_snap, W_snap, n, m, p)
        if snap_error < 1e-8:
            return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            
    return None

def compute_error(U, V, W, n, m, p):
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    error = np.sum((T - 1.0)**2)
    return error
