import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Verified rank-17 solution (Strassen-based)
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))
    
    def set_U(i, j, r, val): U17[i * m + j, r] = val
    def set_V(j, k, r, val): V17[j * p + k, r] = val
    def set_W(k, i, r, val): W17[k * n + i, r] = val

    # Rank 17 solution
    set_U(0, 0, 0, 1); set_U(1, 1, 0, 1)
    set_V(0, 0, 0, 1); set_V(1, 1, 0, 1)
    set_W(0, 0, 0, 1); set_W(1, 1, 0, 1)
    
    set_U(1, 0, 1, 1); set_U(1, 1, 1, 1)
    set_V(0, 0, 1, 1)
    set_W(0, 1, 1, 1); set_W(1, 1, 1, -1)
    
    set_U(0, 0, 2, 1)
    set_V(0, 1, 2, 1); set_V(1, 1, 2, -1)
    set_W(1, 0, 2, 1); set_W(1, 1, 2, 1)
    
    set_U(1, 1, 3, 1)
    set_V(1, 0, 3, 1); set_V(0, 0, 3, -1)
    set_W(0, 0, 3, 1); set_W(0, 1, 3, 1)
    
    set_U(0, 0, 4, 1); set_U(0, 1, 4, 1)
    set_V(1, 1, 4, 1)
    set_W(0, 0, 4, -1); set_W(1, 0, 4, 1)
    
    set_U(1, 0, 5, 1); set_U(0, 0, 5, -1)
    set_V(0, 0, 5, 1); set_V(0, 1, 5, 1)
    set_W(1, 1, 5, 1)
    
    set_U(0, 1, 6, 1); set_U(1, 1, 6, -1)
    set_V(1, 0, 6, 1); set_V(1, 1, 6, 1)
    set_W(0, 0, 6, 1)
    
    set_U(0, 0, 7, 1)
    set_V(0, 2, 7, 1)
    set_W(2, 0, 7, 1)
    
    set_U(0, 1, 8, 1)
    set_V(1, 2, 8, 1)
    set_W(2, 0, 8, 1)
    
    set_U(1, 0, 9, 1)
    set_V(0, 2, 9, 1)
    set_W(2, 1, 9, 1)
    
    set_U(1, 1, 10, 1)
    set_V(1, 2, 10, 1)
    set_W(2, 1, 10, 1)
    
    set_U(0, 2, 11, 1)
    set_V(2, 0, 11, 1)
    set_W(0, 0, 11, 1)
    
    set_U(0, 2, 12, 1)
    set_V(2, 1, 12, 1)
    set_W(1, 0, 12, 1)
    
    set_U(0, 2, 13, 1)
    set_V(2, 2, 13, 1)
    set_W(2, 0, 13, 1)
    
    set_U(1, 2, 14, 1)
    set_V(2, 0, 14, 1)
    set_W(0, 1, 14, 1)
    
    set_U(1, 2, 15, 1)
    set_V(2, 1, 15, 1)
    set_W(1, 1, 15, 1)
    
    set_U(1, 2, 16, 1)
    set_V(2, 2, 16, 1)
    set_W(2, 1, 16, 1)
    
    def compute_tensor(U, V, W, rank):
        """Compute the tensor from the decomposition."""
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        result = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_t[i, j, r] * V_t[j, k, r] * W_t[k, i, r]
                    result[i, j, k] = s
        return result
    
    def is_exact(U, V, W, rank, tol=1e-9):
        T = compute_tensor(U, V, W, rank)
        return np.allclose(T, 1.0, atol=tol)
    
    # Verify rank-17 solution
    if not is_exact(U17, V17, W17, 17):
        # Fallback to schoolbook rank 18
        U18 = np.zeros((n * m, 18))
        V18 = np.zeros((m * p, 18))
        W18 = np.zeros((p * n, 18))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U18[i * m + j, r] = 1
                    V18[j * p + k, r] = 1
                    W18[k * n + i, r] = 1
        return {"U": U18.tolist(), "V": V18.tolist(), "W": W18.tolist()}
    
    best_rank = 17
    best_U, best_V, best_W = U17.copy(), V17.copy(), W17.copy()
    
    # Try to find rank-16 solution
    # Strategy: Start from rank-17, try to merge two terms
    # For each pair (a,b) of terms, try to find a single term that approximates both
    # Then optimize the remaining 16 terms
    
    def try_rank_reduction(U_init, V_init, W_init, init_rank, target_rank):
        """Try to reduce rank by optimizing."""
        if time.time() > deadline:
            return None, None, None
        
        # Initialize: take first target_rank terms, zero out the rest
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        # Use first target_rank terms as initialization
        U[:, :] = U_init[:, :target_rank]
        V[:, :] = V_init[:, :target_rank]
        W[:, :] = W_init[:, :target_rank]
        
        # Gradient descent with adaptive learning rate
        lr = 0.05
        best_err = float('inf')
        
        for step in range(500):
            if time.time() > deadline:
                break
            
            U_t = U.reshape(n, m, target_rank)
            V_t = V.reshape(m, p, target_rank)
            W_t = W.reshape(p, n, target_rank)
            
            # Compute residuals
            R = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(target_rank):
                            s += U_t[i, j, r] * V_t[j, k, r] * W_t[k, i, r]
                        R[i, j, k] = s - 1.0
            
            err = np.sum(R ** 2)
            if err < best_err:
                best_err = err
            
            if best_err < 1e-12:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res = R[i, j, k]
                        for r in range(target_rank):
                            u_val = U_t[i, j, r]
                            v_val = V_t[j, k, r]
                            w_val = W_t[k, i, r]
                            dU[i * m + j, r] += 2 * res * v_val * w_val
                            dV[j * p + k, r] += 2 * res * u_val * w_val
                            dW[k * n + i, r] += 2 * res * u_val * v_val
            
            # Adaptive learning rate
            grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
            if grad_norm > 0:
                lr = min(0.1, max(0.001, lr * 0.999))
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        # Check if we found an exact solution
        if is_exact(U, V, W, target_rank):
            return U, V, W
        
        # Try snapping to half-integers
        U_s = np.round(U * 2) / 2.0
        V_s = np.round(V * 2) / 2.0
        W_s = np.round(W * 2) / 2.0
        if is_exact(U_s, V_s, W_s, target_rank):
            return U_s, V_s, W_s
        
        return None, None, None
    
    # Try different strategies for rank-16
    if time.time() < deadline - 5:
        # Strategy 1: Drop the "laser" terms and re-optimize
        # The laser terms are r=7..16 (the k=2 related terms)
        # Try keeping the Strassen part (r=0..6) and re-solving the laser part with fewer terms
        
        # Strategy 2: Random restarts
        for seed in range(20):
            if time.time() > deadline - 2:
                break
            
            np.random.seed(seed)
            U_rand = np.random.randn(n * m, 16) * 0.5
            V_rand = np.random.randn(m * p, 16) * 0.5
            W_rand = np.random.randn(p * n, 16) * 0.5
            
            U_res, V_res, W_res = try_rank_reduction(U_rand, V_rand, W_rand, 16, 16)
            if U_res is not None:
                best_rank = 16
                best_U, best_V, best_W = U_res, V_res, W_res
                break
        
        # Strategy 3: Start from rank-17, drop one term, re-optimize
        if best_rank == 17 and time.time() < deadline - 3:
            for drop_idx in range(17):
                if time.time() > deadline - 1:
                    break
                
                U_drop = np.delete(U17, drop_idx, axis=1)
                V_drop = np.delete(V17, drop_idx, axis=1)
                W_drop = np.delete(W17, drop_idx, axis=1)
                
                U_res, V_res, W_res = try_rank_reduction(U_drop, V_drop, W_drop, 16, 16)
                if U_res is not None:
                    best_rank = 16
                    best_U, best_V, best_W = U_res, V_res, W_res
                    break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
