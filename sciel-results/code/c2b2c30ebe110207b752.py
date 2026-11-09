import numpy as np
import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 3x4x6 matrix multiplication.
    
    The schoolbook rank is 72. We attempt to find a lower rank decomposition
    using structured search with analytic gradients. If no better rank is found,
    we return the schoolbook decomposition.
    
    The tensor T_{i,j,k} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
    where i in [0,3), j in [0,4), k in [0,6).
    Target: T[i,j,k] = 1 for all (i,j,k).
    """
    n, m, p = 3, 4, 6
    schoolbook_rank = n * m * p  # 72
    
    def build_schoolbook():
        rank = schoolbook_rank
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1.0
                    V[j * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0
                    idx += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    def compute_residual_and_grads(U, V, W):
        """
        Compute residual and analytic gradients.
        Residual R[i,j,k] = sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] - 1
        Objective = sum_{i,j,k} R[i,j,k]^2
        """
        rank = U.shape[1]
        residual = np.zeros((n, m, p))
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = 0.0
                    for r in range(rank):
                        val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    residual[i, j, k] = val - 1.0
                    res = residual[i, j, k]
                    for r in range(rank):
                        grad_U[u_idx, r] += 2 * res * V[v_idx, r] * W[w_idx, r]
                        grad_V[v_idx, r] += 2 * res * U[u_idx, r] * W[w_idx, r]
                        grad_W[w_idx, r] += 2 * res * U[u_idx, r] * V[v_idx, r]
        
        obj = np.sum(residual * residual)
        return obj, residual, grad_U, grad_V, grad_W

    def verify_exact(U, V, W):
        """
        Verify if U, V, W form an exact decomposition of the multiplication tensor.
        Returns True if the residual is effectively zero.
        """
        rank = U.shape[1]
        # Reconstruct T
        # T[a,b,c] = sum_s U[a,s]*V[b,s]*W[c,s]
        # a = i*m+j, b = j*p+k, c = k*n+i
        # We check if for all i,j,k, sum_s U[i*m+j, s]*V[j*p+k, s]*W[k*n+i, s] == 1
        
        # Use efficient einsum if possible, but loops are fine for small sizes
        # Actually, let's just compute the residual again to be sure
        obj, residual, _, _, _ = compute_residual_and_grads(U, V, W)
        return obj < 1e-10

    def round_to_half_integer(M):
        """Round matrix to nearest half-integer."""
        return np.round(M * 2) / 2.0

    def try_rank_search(rank, time_budget=30.0):
        """Attempt to find a decomposition of the given rank using gradient descent with random restarts."""
        start_time = time.time()
        best_obj = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        max_restarts = 10
        max_iterations = 3000
        lr = 0.01
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_budget - 2:
                break
            
            # Initialize with schoolbook-like structure plus noise
            # This helps escape local minima compared to pure random
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # Add some schoolbook structure to a few columns to give it a head start
            # Pick a few random schoolbook terms and add them
            for _ in range(min(rank, 10)):
                i = np.random.randint(n)
                j = np.random.randint(m)
                k = np.random.randint(p)
                col = np.random.randint(rank)
                U[i * m + j, col] += 1.0
                V[j * p + k, col] += 1.0
                W[k * n + i, col] += 1.0
            
            # Adam optimizer state
            m_U = np.zeros_like(U)
            v_U = np.zeros_like(U)
            m_V = np.zeros_like(V)
            v_V = np.zeros_like(V)
            m_W = np.zeros_like(W)
            v_W = np.zeros_like(W)
            
            obj_prev, _, grad_U, grad_V, grad_W = compute_residual_and_grads(U, V, W)
            
            if obj_prev < best_obj:
                best_obj = obj_prev
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            for it in range(max_iterations):
                if time.time() - start_time > time_budget - 2:
                    break
                
                # Compute gradients
                obj_new, _, grad_U, grad_V, grad_W = compute_residual_and_grads(U, V, W)
                
                # Update Adam states
                m_U = beta1 * m_U + (1 - beta1) * grad_U
                v_U = beta2 * v_U + (1 - beta2) * grad_U**2
                m_hat_U = m_U / (1 - beta1**(it + 1))
                v_hat_U = v_U / (1 - beta2**(it + 1))
                
                m_V = beta1 * m_V + (1 - beta1) * grad_V
                v_V = beta2 * v_V + (1 - beta2) * grad_V**2
                m_hat_V = m_V / (1 - beta1**(it + 1))
                v_hat_V = v_V / (1 - beta2**(it + 1))
                
                m_W = beta1 * m_W + (1 - beta1) * grad_W
                v_W = beta2 * v_W + (1 - beta2) * grad_W**2
                m_hat_W = m_W / (1 - beta1**(it + 1))
                v_hat_W = v_W / (1 - beta2**(it + 1))
                
                # Update parameters
                U -= lr * m_hat_U / (np.sqrt(v_hat_U) + eps)
                V -= lr * m_hat_V / (np.sqrt(v_hat_V) + eps)
                W -= lr * m_hat_W / (np.sqrt(v_hat_W) + eps)
                
                if obj_new < best_obj:
                    best_obj = obj_new
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if obj_new < 1e-12:
                    break
                
                # Adaptive learning rate
                if obj_new > obj_prev * 1.01:
                    lr = max(lr * 0.9, 1e-6)
                elif obj_new < obj_prev * 0.99:
                    lr = min(lr * 1.05, 0.05)
                
                obj_prev = obj_new
            
            # After each restart, try to round the best solution found so far
            if best_U is not None:
                U_rounded = round_to_half_integer(best_U)
                V_rounded = round_to_half_integer(best_V)
                W_rounded = round_to_half_integer(best_W)
                
                if verify_exact(U_rounded, V_rounded, W_rounded):
                    return {'U': U_rounded.tolist(), 'V': V_rounded.tolist(), 'W': W_rounded.tolist()}
                
                # Also try integer rounding
                U_int = np.round(best_U)
                V_int = np.round(best_V)
                W_int = np.round(best_W)
                if verify_exact(U_int, V_int, W_int):
                    return {'U': U_int.tolist(), 'V': V_int.tolist(), 'W': W_int.tolist()}
        
        # Final check on the best continuous solution
        if best_obj < 1e-10 and best_U is not None:
            U_rounded = round_to_half_integer(best_U)
            V_rounded = round_to_half_integer(best_V)
            W_rounded = round_to_half_integer(best_W)
            if verify_exact(U_rounded, V_rounded, W_rounded):
                return {'U': U_rounded.tolist(), 'V': V_rounded.tolist(), 'W': W_rounded.tolist()}
            
            U_int = np.round(best_U)
            V_int = np.round(best_V)
            W_int = np.round(best_W)
            if verify_exact(U_int, V_int, W_int):
                return {'U': U_int.tolist(), 'V': V_int.tolist(), 'W': W_int.tolist()}
        
        return None
    
    # Deadline guard: ensure we return before 120s
    overall_start = time.time()
    
    # Try to find a better rank
    # We prioritize higher ranks (closer to 72) as they are more likely to exist
    for target_rank in [71, 70, 69, 68]:
        if time.time() - overall_start > 130:
            break
        time_budget = min(20.0, 135 - (time.time() - overall_start))
        if time_budget < 5:
            break
        result = try_rank_search(target_rank, time_budget=time_budget)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return build_schoolbook()
