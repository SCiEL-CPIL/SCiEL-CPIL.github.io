import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor indices
    # U: (n*m, R) -> i*m + j
    # V: (m*p, R) -> j*p + k
    # W: (p*n, R) -> k*n + i
    
    size_U = n * m
    size_V = m * p
    size_W = p * n
    
    # Build target tensor T as a flat vector
    # T[a, b, c] = 1 if a=(i,j), b=(j',k'), c=(k'',i'') and j==j' and k'==k'' and i''==i
    # Actually, the tensor is defined such that the product U V W reconstructs it.
    # We want sum_r U_{a,r} V_{b,r} W_{c,r} = T_{a,b,c}
    
    T_flat = np.zeros(size_U * size_V * size_W)
    
    # Iterate over all valid (i, j, k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_U = i * m + j
                idx_V = j * p + k
                idx_W = k * n + i
                # Flatten index: (idx_U, idx_V, idx_W)
                # We need to map this 3D index to 1D
                # T_flat[index] = 1
                # index = idx_U * (size_V * size_W) + idx_V * size_W + idx_W
                flat_idx = idx_U * (size_V * size_W) + idx_V * size_W + idx_W
                T_flat[flat_idx] = 1.0
                
    T_reshaped = T_flat.reshape(size_U, size_V, size_W)
    
    def residual_func(params, rank):
        # params are flattened U, V, W
        u_size = size_U * rank
        v_size = size_V * rank
        w_size = size_W * rank
        
        u_flat = params[:u_size]
        v_flat = params[u_size:u_size+v_size]
        w_flat = params[u_size+v_size:]
        
        U = u_flat.reshape(size_U, rank)
        V = v_flat.reshape(size_V, rank)
        W = w_flat.reshape(size_W, rank)
        
        # Compute tensor T_est[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # This is equivalent to:
        # T_est = np.einsum('ar,br,cr->abc', U, V, W)
        T_est = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Flatten and return difference
        return (T_est - T_reshaped).flatten()
    
    def try_rank(rank, seeds=20):
        for seed in range(seeds):
            if time.time() > 140: # Safety margin
                return None
            
            np.random.seed(seed)
            # Random initialization
            U_init = np.random.randn(size_U, rank)
            V_init = np.random.randn(size_V, rank)
            W_init = np.random.randn(size_W, rank)
            
            # Scale to help optimization
            # Normalize columns?
            for col in range(rank):
                if np.linalg.norm(U_init[:, col]) > 0:
                    U_init[:, col] /= np.linalg.norm(U_init[:, col])
                if np.linalg.norm(V_init[:, col]) > 0:
                    V_init[:, col] /= np.linalg.norm(V_init[:, col])
                if np.linalg.norm(W_init[:, col]) > 0:
                    W_init[:, col] /= np.linalg.norm(W_init[:, col])
            
            init_params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = least_squares(residual_func, init_params, args=(rank,), method='lm', max_nfev=2000)
            except:
                continue
                
            if res.success or res.cost < 1e-10:
                # Try to snap to integers or half-integers
                u_size = size_U * rank
                v_size = size_V * rank
                w_size = size_W * rank
                
                U_sol = res.x[:u_size].reshape(size_U, rank)
                V_sol = res.x[u_size:u_size+v_size].reshape(size_V, rank)
                W_sol = res.x[u_size+v_size:].reshape(size_W, rank)
                
                # Snap function: nearest 0.5
                def snap(x):
                    return np.round(x * 2) / 2
                
                U_snap = snap(U_sol)
                V_snap = snap(V_sol)
                W_snap = snap(W_sol)
                
                # Verify exact match
                T_est_snap = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
                if np.allclose(T_est_snap, T_reshaped, atol=1e-9):
                    # Found a valid decomposition
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
                    
        return None

    # Try ranks from low to high
    # Lower bound is likely > 15. Try 16, 17, 18.
    for rank in [16, 17, 18]:
        result = try_rank(rank, seeds=50)
        if result:
            return result
            
    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
