import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit
    
    # Target tensor T[i, j, k] = 1
    # We want sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1
    
    best_rank = 17
    best_U, best_V, best_W = None, None, None
    
    # Store the current rank 17 solution as fallback
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))
    
    def set_U17(i, j, r, val): U17[i * m + j, r] = val
    def set_V17(j, k, r, val): V17[j * p + k, r] = val
    def set_W17(k, i, r, val): W17[k * n + i, r] = val

    # Rank 17 solution: Strassen 2x2x2 for (i in {0,1}, j in {0,1}, k in {0,1})
    # + direct terms for remaining blocks
    set_U17(0, 0, 0, 1); set_U17(1, 1, 0, 1)
    set_V17(0, 0, 0, 1); set_V17(1, 1, 0, 1)
    set_W17(0, 0, 0, 1); set_W17(1, 1, 0, 1)
    
    set_U17(1, 0, 1, 1); set_U17(1, 1, 1, 1)
    set_V17(0, 0, 1, 1)
    set_W17(0, 1, 1, 1); set_W17(1, 1, 1, -1)
    
    set_U17(0, 0, 2, 1)
    set_V17(0, 1, 2, 1); set_V17(1, 1, 2, -1)
    set_W17(1, 0, 2, 1); set_W17(1, 1, 2, 1)
    
    set_U17(1, 1, 3, 1)
    set_V17(1, 0, 3, 1); set_V17(0, 0, 3, -1)
    set_W17(0, 0, 3, 1); set_W17(0, 1, 3, 1)
    
    set_U17(0, 0, 4, 1); set_U17(0, 1, 4, 1)
    set_V17(1, 1, 4, 1)
    set_W17(0, 0, 4, -1); set_W17(1, 0, 4, 1)
    
    set_U17(1, 0, 5, 1); set_U17(0, 0, 5, -1)
    set_V17(0, 0, 5, 1); set_V17(0, 1, 5, 1)
    set_W17(1, 1, 5, 1)
    
    set_U17(0, 1, 6, 1); set_U17(1, 1, 6, -1)
    set_V17(1, 0, 6, 1); set_V17(1, 1, 6, 1)
    set_W17(0, 0, 6, 1)
    
    # Direct terms for j in {0,1}, k=2
    set_U17(0, 0, 7, 1); set_V17(0, 2, 7, 1); set_W17(2, 0, 7, 1)
    set_U17(0, 1, 8, 1); set_V17(1, 2, 8, 1); set_W17(2, 0, 8, 1)
    set_U17(1, 0, 9, 1); set_V17(0, 2, 9, 1); set_W17(2, 1, 9, 1)
    set_U17(1, 1, 10, 1); set_V17(1, 2, 10, 1); set_W17(2, 1, 10, 1)
    
    # Direct terms for j=2, k in {0,1,2}
    set_U17(0, 2, 11, 1); set_V17(2, 0, 11, 1); set_W17(0, 0, 11, 1)
    set_U17(0, 2, 12, 1); set_V17(2, 1, 12, 1); set_W17(1, 0, 12, 1)
    set_U17(0, 2, 13, 1); set_V17(2, 2, 13, 1); set_W17(2, 0, 13, 1)
    set_U17(1, 2, 14, 1); set_V17(2, 0, 14, 1); set_W17(0, 1, 14, 1)
    set_U17(1, 2, 15, 1); set_V17(2, 1, 15, 1); set_W17(1, 1, 15, 1)
    set_U17(1, 2, 16, 1); set_V17(2, 2, 16, 1); set_W17(2, 1, 16, 1)
    
    best_U, best_V, best_W = U17, V17, W17

    def compute_error(U, V, W, rank):
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_t[i, j, r] * V_t[j, k, r] * W_t[k, i, r]
                    error += (s - 1.0) ** 2
        return error

    def is_exact(U, V, W, rank):
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_t[i, j, r] * V_t[j, k, r] * W_t[k, i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def snap_to_half_integers(U, V, W):
        U_s = np.round(U * 2) / 2.0
        V_s = np.round(V * 2) / 2.0
        W_s = np.round(W * 2) / 2.0
        return U_s, V_s, W_s

    # Strategy 1: Try to merge pairs of simple terms using Strassen-like tricks
    # The simple terms are r=7..16 (indices 7-16 in the rank-17 solution)
    # Each has exactly one non-zero in U, one in V, one in W
    # We can try to combine two such terms if they share structure
    
    # Identify the simple terms and their (i,j), (j,k), (k,i) indices
    simple_terms = []
    for r in range(7, 17):
        u_idx = np.where(U17[:, r] != 0)[0]
        v_idx = np.where(V17[:, r] != 0)[0]
        w_idx = np.where(W17[:, r] != 0)[0]
        if len(u_idx) == 1 and len(v_idx) == 1 and len(w_idx) == 1:
            u_i, u_j = u_idx[0] // m, u_idx[0] % m
            v_j, v_k = v_idx[0] // p, v_idx[0] % p
            w_k, w_i = w_idx[0] // n, w_idx[0] % n
            simple_terms.append((r, (u_i, u_j), (v_j, v_k), (w_k, w_i)))
    
    # Try to find if any two simple terms can be merged
    # Two terms can be merged if we can find a 2x2 Strassen-like combination
    # Term A: U[a_i,a_j] V[b_j,b_k] W[c_k,c_i]
    # Term B: U[d_i,d_j] V[e_j,e_k] W[f_k,f_i]
    # If they share some indices, we might be able to combine
    
    # More practical approach: use gradient descent starting from rank-17
    # with one term removed, and see if we can converge to an exact rank-16
    
    def try_rank16_from_rank17(remove_idx):
        """Try to find rank-16 by removing term 'remove_idx' from rank-17 and optimizing"""
        if time.time() > deadline:
            return None, None, None
        
        # Create initial rank-16 by removing the specified term
        keep = [r for r in range(17) if r != remove_idx]
        U16 = U17[:, keep].copy()
        V16 = V17[:, keep].copy()
        W16 = W17[:, keep].copy()
        
        # Gradient descent to try to make it exact
        lr = 0.005
        best_err = float('inf')
        
        for step in range(3000):
            if time.time() > deadline:
                break
            
            U_t = U16.reshape(n, m, 16)
            V_t = V16.reshape(m, p, 16)
            W_t = W16.reshape(p, n, 16)
            
            # Compute residuals and gradients
            R = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(16):
                            s += U_t[i, j, r] * V_t[j, k, r] * W_t[k, i, r]
                        R[i, j, k] = s - 1.0
            
            dU = np.zeros_like(U16)
            dV = np.zeros_like(V16)
            dW = np.zeros_like(W16)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res = R[i, j, k]
                        for r in range(16):
                            u_val = U_t[i, j, r]
                            v_val = V_t[j, k, r]
                            w_val = W_t[k, i, r]
                            dU[i * m + j, r] += 2 * res * v_val * w_val
                            dV[j * p + k, r] += 2 * res * u_val * w_val
                            dW[k * n + i, r] += 2 * res * u_val * v_val
            
            U16 -= lr * dU
            V16 -= lr * dV
            W16 -= lr * dW
            
            # Periodic snap
            if step % 200 == 0:
                U16, V16, W16 = snap_to_half_integers(U16, V16, W16)
            
            # Check convergence
            err = compute_error(U16, V16, W16, 16)
            if err < best_err:
                best_err = err
            
            if is_exact(U16, V16, W16, 16):
                return U16, V16, W16
        
        # Check if close to exact
        if best_err < 1e-6:
            U16, V16, W16 = snap_to_half_integers(U16, V16, W16)
            if is_exact(U16, V16, W16, 16):
                return U16, V16, W16
        
        return None, None, None

    # Try removing each of the "simple" terms (r=7..16)
    for remove_idx in range(7, 17):
        if time.time() > deadline:
            break
        
        U16, V16, W16 = try_rank16_from_rank17(remove_idx)
        if U16 is not None:
            best_rank = 16
            best_U, best_V, best_W = U16, V16, W16
            break
    
    # Strategy 2: Random initialization with gradient descent for rank 16
    if best_rank > 16:
        for seed in range(30):
            if time.time() > deadline:
                break
            
            np.random.seed(seed)
            
            U = np.random.randn(n * m, 16) * 0.3
            V = np.random.randn(m * p, 16) * 0.3
            W = np.random.randn(p * n, 16) * 0.3
            
            lr = 0.01
            for step in range(2000):
                if time.time() > deadline:
                    break
                
                U_t = U.reshape(n, m, 16)
                V_t = V.reshape(m, p, 16)
                W_t = W.reshape(p, n, 16)
                
                R = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(16):
                                s += U_t[i, j, r] * V_t[j, k, r] * W_t[k, i, r]
                            R[i, j, k] = s - 1.0
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            res = R[i, j, k]
                            for r in range(16):
                                u_val = U_t[i, j, r]
                                v_val = V_t[j, k, r]
                                w_val = W_t[k, i, r]
                                dU[i * m + j, r] += 2 * res * v_val * w_val
                                dV[j * p + k, r] += 2 * res * u_val * w_val
                                dW[k * n + i, r] += 2 * res * u_val * v_val
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                if step % 150 == 0:
                    U, V, W = snap_to_half_integers(U, V, W)
                
                if is_exact(U, V, W, 16):
                    best_rank = 16
                    best_U, best_V, best_W = U, V, W
                    break
            
            if best_rank == 16:
                break

    # Final verification
    if best_U is not None and best_V is not None and best_W is not None:
        if not is_exact(best_U, best_V, best_W, best_rank):
            # Fallback to known rank-17
            best_rank = 17
            best_U, best_V, best_W = U17, V17, W17

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
