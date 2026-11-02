import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i) are consistent, else 0
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build target tensor
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target[u, v, w] = 1.0

    # Flatten target for optimization
    target_flat = target.flatten()
    
    def unpack_and_compute(U_flat, V_flat, W_flat, rank):
        U = U_flat.reshape(nm, rank)
        V = V_flat.reshape(mp, rank)
        W = W_flat.reshape(pn, rank)
        # Compute tensor sum
        # T[u, v, w] = sum_c U[u, c] V[v, c] W[w, c]
        # This is a tensor contraction.
        # Efficient way:
        # Reshape U to (nm, rank), V to (mp, rank), W to (pn, rank)
        # We want T[u, v, w]
        # T = sum_c U[:, c] * V[:, c] * W[:, c] (outer product)
        # Can be done via einsum
        T = np.einsum('uc,vc,wc->uvw', U, V, W)
        return T.flatten()

    def objective(params):
        rank = len(params) // 3
        U_flat = params[:rank * nm]
        V_flat = params[rank * nm : 2 * rank * nm]
        W_flat = params[2 * rank * nm : 3 * rank * nm]
        
        T_flat = unpack_and_compute(U_flat, V_flat, W_flat, rank)
        return T_flat - target_flat

    # Start with schoolbook
    current_rank = nm * mp * pn
    best_U = np.zeros((nm, current_rank))
    best_V = np.zeros((mp, current_rank))
    best_W = np.zeros((pn, current_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                best_U[u, col] = 1.0
                best_V[v, col] = 1.0
                best_W[w, col] = 1.0
                col += 1
                
    best_params = np.concatenate([best_U.flatten(), best_V.flatten(), best_W.flatten()])
    best_residual = 0.0
    
    # Try to reduce rank
    for target_rank in range(current_rank - 1, 40, -1):
        if time.time() > deadline:
            break
            
        # Initialize with truncated best params
        # We need to map the old rank to the new rank.
        # Simple strategy: take the first target_rank columns of the best solution.
        # Or better, random restarts?
        # Let's try truncating the best known solution.
        
        init_U = best_U[:, :target_rank]
        init_V = best_V[:, :target_rank]
        init_W = best_W[:, :target_rank]
        
        init_params = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
        
        try:
            # Use least_squares
            # Method 'lm' is fast but might get stuck. 'trf' is more robust.
            # Given the small size, 'lm' might be sufficient for small reductions.
            res = least_squares(objective, init_params, method='lm', max_nfev=500)
            
            if res.cost < best_residual * 0.1 + 1e-10: # Significant improvement
                new_params = res.x
                new_U = new_params[:target_rank * nm].reshape(nm, target_rank)
                new_V = new_params[target_rank * nm : 2 * target_rank * nm].reshape(mp, target_rank)
                new_W = new_params[2 * target_rank * nm : 3 * target_rank * nm].reshape(pn, target_rank)
                
                best_U = new_U
                best_V = new_V
                best_W = new_W
                best_params = new_params
                best_residual = res.cost
                current_rank = target_rank
        except Exception:
            pass

    # Return the best found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
