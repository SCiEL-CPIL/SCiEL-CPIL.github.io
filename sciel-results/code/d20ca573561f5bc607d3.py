"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Build the target tensor T
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some valid i,j,k
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    # Schoolbook decomposition as fallback
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
    
    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    def verify_and_snap(U, V, W, r):
        """Verify the decomposition and snap to half-integers."""
        # Try snapping to half-integers
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        # Reconstruct
        T_recon = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
        
        if np.allclose(T_recon, T_target, atol=1e-9):
            return U_snap, V_snap, W_snap
        
        # Try snapping to integers
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        
        T_recon_int = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
        
        if np.allclose(T_recon_int, T_target, atol=1e-9):
            return U_int, V_int, W_int
        
        # Use the original values
        T_recon_orig = np.einsum('ar,br,cr->abc', U, V, W)
        
        if np.allclose(T_recon_orig, T_target, atol=1e-8):
            return U, V, W
        
        return None
    
    def try_rank(r, seed=42, max_iter=200):
        """Try to find a rank-r decomposition."""
        if time.time() > deadline - 5:
            return None
        
        np.random.seed(seed)
        
        # Initialize with random values
        U = np.random.randn(n * m, r) * 0.5
        V = np.random.randn(m * p, r) * 0.5
        W = np.random.randn(p * n, r) * 0.5
        
        size_U = (n * m) * r
        size_V = (m * p) * r
        size_W = (p * n) * r
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def loss(params):
            U_flat = params[:size_U]
            V_flat = params[size_U:size_U + size_V]
            W_flat = params[size_U + size_V:]
            
            U_p = U_flat.reshape((n * m), r)
            V_p = V_flat.reshape((m * p), r)
            W_p = W_flat.reshape((p * n), r)
            
            T_recon = np.einsum('ar,br,cr->abc', U_p, V_p, W_p)
            err = np.sum((T_recon - T_target) ** 2)
            return err
        
        def loss_grad(params):
            # Compute gradient analytically
            U_flat = params[:size_U]
            V_flat = params[size_U:size_U + size_V]
            W_flat = params[size_U + size_V:]
            
            U_p = U_flat.reshape((n * m), r)
            V_p = V_flat.reshape((m * p), r)
            W_p = W_flat.reshape((p * n), r)
            
            T_recon = np.einsum('ar,br,cr->abc', U_p, V_p, W_p)
            diff = T_recon - T_target
            
            # Gradient with respect to U
            # d/dU[a,r] = sum_{b,c} diff[a,b,c] * V_p[b,r] * W_p[c,r]
            grad_U = np.einsum('abc,br,cr->ar', diff, V_p, W_p)
            
            # Gradient with respect to V
            # d/dV[b,r] = sum_{a,c} diff[a,b,c] * U_p[a,r] * W_p[c,r]
            grad_V = np.einsum('abc,ar,cr->br', diff, U_p, W_p)
            
            # Gradient with respect to W
            # d/dW[c,r] = sum_{a,b} diff[a,b,c] * U_p[a,r] * V_p[b,r]
            grad_W = np.einsum('abc,ar,br->cr', diff, U_p, V_p)
            
            return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        try:
            result = minimize(loss, x0, method='L-BFGS-B', jac=loss_grad,
                            options={'maxiter': max_iter, 'maxfun': 500, 'gtol': 1e-8})
            
            if result.fun < 1e-6:
                x_opt = result.x
                U_opt = x_opt[:size_U].reshape((n * m), r)
                V_opt = x_opt[size_U:size_U + size_V].reshape((m * p), r)
                W_opt = x_opt[size_U + size_V:].reshape((p * n), r)
                
                snapped = verify_and_snap(U_opt, V_opt, W_opt, r)
                if snapped:
                    return snapped
        except Exception:
            pass
        
        return None
    
    # Try various ranks from low to high
    # Based on known results, try ranks around 20-40
    ranks_to_try = [20, 25, 30, 35, 40, 45]
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
        
        found = False
        max_attempts = 5
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            seed = 42 + attempt * 1000
            result = try_rank(target_rank, seed=seed)
            
            if result:
                U_sol, V_sol, W_sol = result
                best_rank = target_rank
                best_U = U_sol
                best_V = V_sol
                best_W = W_sol
                found = True
                break
        
        if found:
            # Try to go lower
            continue
    
    # Convert to lists for return
    best_U_list = best_U.tolist()
    best_V_list = best_V.tolist()
    best_W_list = best_W.tolist()
    
    return {"U": best_U_list, "V": best_V_list, "W": best_W_list}
# EVOLVE-BLOCK-END
