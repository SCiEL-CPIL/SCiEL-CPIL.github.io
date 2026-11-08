import numpy as np
import time
import sys
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Target rank
    target_rank = 17
    
    # Precompute non-zero indices of the tensor T
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    # u in [0, nm), v in [0, mp), w in [0, pn)
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                nz_indices.append((u, v, w))
    
    # All indices for zero constraints
    all_u = np.arange(n * m)
    all_v = np.arange(m * p)
    all_w = np.arange(p * n)
    # We only need to check the non-zero ones for equality to 1, and the rest for equality to 0.
    # But checking all 324 is fast enough.
    
    def error_func(params):
        # params is flattened U, V, W
        size_U = n * m * target_rank
        size_V = m * p * target_rank
        size_W = p * n * target_rank
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:]
        
        U = U_flat.reshape(n * m, target_rank)
        V = V_flat.reshape(m * p, target_rank)
        W = W_flat.reshape(p * n, target_rank)
        
        # Compute reconstruction T_recon[u, v, w] = sum_r U[u,r] V[v,r] W[w,r]
        # This is a tensor product.
        # We can compute this efficiently using einsum or loops.
        # T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
        # But we only care about the error.
        # Error = sum (T_recon - T_target)^2
        
        # To save time, we can just compute the error on the non-zero and zero entries.
        # Actually, computing the full tensor is 6*9*6*17 ~ 5k operations, very fast.
        T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
        
        # Target tensor
        # T_target is 0 everywhere except nz_indices where it is 1.
        # We can create a target tensor of zeros and set nz to 1.
        # But doing this every call is wasteful. 
        # Instead, we can calculate the error as:
        # sum_all T_recon^2 - 2 * sum_nz T_recon[nz] + count_nz
        # Because T_target is 1 at nz, 0 elsewhere.
        # (T_recon - T_target)^2 = T_recon^2 - 2 T_recon T_target + T_target^2
        # Sum = sum(T_recon^2) - 2 * sum(T_recon[nz]) + len(nz)
        
        sum_sq = np.sum(T_recon**2)
        sum_nz_val = 0
        for u, v, w in nz_indices:
            sum_nz_val += T_recon[u, v, w]
            
        error = sum_sq - 2 * sum_nz_val + len(nz_indices)
        return error

    def gradient_func(params):
        # Numerical gradient is slow. Analytical is better.
        # dE/dU[u,r] = 2 * sum_{v,w} (T_recon[u,v,w] - T_target[u,v,w]) V[v,r] W[w,r]
        # This requires storing T_recon and T_target.
        # For simplicity and given small size, we can use finite differences or just let scipy use numerical if we don't provide jac.
        # But L-BFGS with numerical gradient might be slow.
        # Let's try providing an analytical gradient.
        
        size_U = n * m * target_rank
        size_V = m * p * target_rank
        size_W = p * n * target_rank
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:]
        
        U = U_flat.reshape(n * m, target_rank)
        V = V_flat.reshape(m * p, target_rank)
        W = W_flat.reshape(p * n, target_rank)
        
        T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
        
        # Create T_target (sparse)
        # We need (T_recon - T_target)
        # Let's create a full T_target array once outside? No, it's constant.
        # T_target is 0 except at nz_indices.
        
        # Gradient wrt U:
        # dE/dU = 2 * sum_{v,w} (T_recon - T_target)[:, v, w] * V[v, :] * W[w, :]
        # This is like a tensor contraction.
        # diff = T_recon - T_target
        # dU = 2 * np.einsum('uvw,vr,wr->ur', diff, V, W)
        
        # We need T_target.
        T_target = np.zeros((n*m, m*p, p*n))
        for u, v, w in nz_indices:
            T_target[u, v, w] = 1.0
            
        diff = T_recon - T_target
        
        dU = 2 * np.einsum('uvw,vr,wr->ur', diff, V, W)
        dV = 2 * np.einsum('uvw,ur,wr->vr', diff, U, W)
        dW = 2 * np.einsum('uvw,ur,vr->wr', diff, U, V)
        
        return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])

    # Optimization
    start_time = time.time()
    deadline = start_time + 140
    
    best_solution = None
    best_error = float('inf')
    
    # Try a few random restarts
    num_restarts = 5
    rng = np.random.default_rng(42)
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize
        # Scale: U, V, W entries should be around 1.
        U_init = rng.standard_normal((n * m, target_rank))
        V_init = rng.standard_normal((m * p, target_rank))
        W_init = rng.standard_normal((p * n, target_rank))
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            res = minimize(
                error_func, 
                params_init, 
                jac=gradient_func, 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < best_error:
                best_error = res.fun
                if res.fun < 1e-6:
                    # Check if we can snap
                    size_U = n * m * target_rank
                    size_V = m * p * target_rank
                    size_W = p * n * target_rank
                    
                    U_opt = res.x[:size_U].reshape(n * m, target_rank)
                    V_opt = res.x[size_U:size_U + size_V].reshape(m * p, target_rank)
                    W_opt = res.x[size_U + size_V:].reshape(p * n, target_rank)
                    
                    # Snap to nearest half-integer
                    U_snap = np.round(U_opt * 2) / 2
                    V_snap = np.round(V_opt * 2) / 2
                    W_snap = np.round(W_opt * 2) / 2
                    
                    # Verify exactness
                    T_recon_snap = np.einsum('ur,vr,wr->uvw', U_snap, V_snap, W_snap)
                    
                    # Check non-zeros
                    valid = True
                    for u, v, w in nz_indices:
                        if abs(T_recon_snap[u, v, w] - 1.0) > 1e-9:
                            valid = False
                            break
                    
                    if valid:
                        # Check zeros (sample or all)
                        # All zeros check is fast
                        T_target_full = np.zeros((n*m, m*p, p*n))
                        for u, v, w in nz_indices:
                            T_target_full[u, v, w] = 1.0
                        if np.allclose(T_recon_snap, T_target_full, atol=1e-9):
                            best_solution = {
                                "U": U_snap.tolist(),
                                "V": V_snap.tolist(),
                                "W": W_snap.tolist()
                            }
                            # Found a solution, break
                            return best_solution
                            
        except Exception as e:
            pass
            # print(f"Error in restart {r}: {e}")
            
    # Fallback to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }

# EVOLVE-BLOCK-END
