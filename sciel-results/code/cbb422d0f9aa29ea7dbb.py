import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Schoolbook decomposition
    def schoolbook_decomposition():
        U = np.zeros((n * m, rank_schoolbook))
        V = np.zeros((m * p, rank_schoolbook))
        W = np.zeros((p * n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Check decomposition validity
    def check_decomposition(decomp, tol=1e-6):
        U = np.array(decomp["U"], dtype=float)
        V = np.array(decomp["V"], dtype=float)
        W = np.array(decomp["W"], dtype=float)
        
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False
        
        R = U.shape[1]
        
        # Check tensor equality
        for i in range(n):
            for ip in range(n):
                for j in range(m):
                    for jp in range(m):
                        for k in range(p):
                            for kp in range(p):
                                expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                val = 0.0
                                for r in range(R):
                                    val += U[i * m + j, r] * V[jp * p + k, r] * W[ip * n + kp, r]
                                if abs(val - expected) > tol:
                                    return False
        return True

    def try_find_decomposition(target_rank, max_iter=2000, time_limit=10.0):
        start_time = time.time()
        
        # Create target tensor
        T_target = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_target[a, b, c] = 1.0
        
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(params):
            U_flat, V_flat, W_flat = np.split(params, [n*m*target_rank, (n*m + m*p)*target_rank])
            U = U_flat.reshape(n*m, target_rank)
            V = V_flat.reshape(m*p, target_rank)
            W = W_flat.reshape(p*n, target_rank)
            
            # Reconstruct tensor
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            
            # Compute error
            diff = T_rec - T_target
            return np.sum(diff * diff)

        from scipy.optimize import minimize
        try:
            res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': max_iter, 'maxfun': 10000})
        except Exception:
            return None
        
        if time.time() - start_time > time_limit:
            return None
            
        if res.fun < 1e-6:
            U_flat, V_flat, W_flat = np.split(res.x, [n*m*target_rank, (n*m + m*p)*target_rank])
            U = U_flat.reshape(n*m, target_rank)
            V = V_flat.reshape(m*p, target_rank)
            W = W_flat.reshape(p*n, target_rank)
            
            # Round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            decomp = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
            if check_decomposition(decomp, tol=1e-5):
                return decomp
            else:
                U_rounded = np.round(U)
                V_rounded = np.round(V)
                W_rounded = np.round(W)
                decomp = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                if check_decomposition(decomp, tol=1e-5):
                    return decomp
        return None

    best_decomp = schoolbook_decomposition()
    best_rank = rank_schoolbook
    
    start_time = time.time()
    # Try to reduce rank
    # Start from 41 down to 30
    for target_rank in range(rank_schoolbook - 1, 29, -1):
        if time.time() - start_time > 100: # Leave 20s margin
            break
        decomp = try_find_decomposition(target_rank)
        if decomp is not None:
            best_decomp = decomp
            best_rank = target_rank
            break
            
    return best_decomp
