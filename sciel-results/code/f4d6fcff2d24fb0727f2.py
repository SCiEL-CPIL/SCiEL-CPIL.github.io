"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import scipy
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    num_u = n * m  # 8
    num_v = m * p  # 28
    num_w = p * n  # 14
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p  # 56
        U = np.zeros((num_u, rank), dtype=float)
        V = np.zeros((num_v, rank), dtype=float)
        W = np.zeros((num_w, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U0, V0, W0, rank0 = schoolbook()
    
    # Validate function
    def validate(U, V, W, rank):
        """Check if the decomposition exactly reconstructs the tensor."""
        T = np.zeros((n, m, p), dtype=float)
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            # Only process non-zero entries for efficiency
            u_nz = np.nonzero(uc)[0]
            v_nz = np.nonzero(vc)[0]
            w_nz = np.nonzero(wc)[0]
            for ui in u_nz:
                i = ui // m
                j = ui % m
                uv = uc[ui]
                for vi in v_nz:
                    vj = vi // p
                    k = vi % p
                    if vj != j:
                        continue
                    vv = vc[vi]
                    for wi in w_nz:
                        wk = wi // n
                        wi_i = wi % n
                        if wk != k or wi_i != i:
                            continue
                        wv = wc[wi]
                        T[i, j, k] += uv * vv * wv
        return np.allclose(T, 1.0, atol=1e-8)
    
    # Loss function for optimization
    def loss(params, rank):
        """Compute loss for given parameters and rank."""
        n_u = num_u * rank
        n_v = num_v * rank
        u_flat = params[:n_u]
        v_flat = params[n_u:n_u + n_v]
        w_flat = params[n_u + n_v:]
        
        U = u_flat.reshape((num_u, rank))
        V = v_flat.reshape((num_v, rank))
        W = w_flat.reshape((num_w, rank))
        
        T = np.zeros((n, m, p), dtype=float)
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            u_nz = np.nonzero(uc)[0]
            v_nz = np.nonzero(vc)[0]
            w_nz = np.nonzero(wc)[0]
            for ui in u_nz:
                i = ui // m
                j = ui % m
                uv = uc[ui]
                for vi in v_nz:
                    vj = vi // p
                    k = vi % p
                    if vj != j:
                        continue
                    vv = vc[vi]
                    for wi in w_nz:
                        wk = wi // n
                        wi_i = wi % n
                        if wk != k or wi_i != i:
                            continue
                        wv = wc[wi]
                        T[i, j, k] += uv * vv * wv
        return np.sum((T - 1.0)**2)
    
    # Try to find a better decomposition using optimization
    # Start with various rank values and optimize
    
    best_U, best_V, best_W, best_rank = U0, V0, W0, rank0
    best_loss = float('inf')
    
    # Try different ranks from a lower bound
    # Lower bound is max(num_u, num_v, num_w) = 28
    # But we know Strassen can help, so try ranks around 40-50 first
    ranks_to_try = []
    for r in range(30, 57):
        ranks_to_try.append(r)
    
    # Shuffle to try different orders
    np.random.shuffle(ranks_to_try)
    
    for rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
        
        # Initialize with random values
        for attempt in range(3):
            if time.time() > deadline - 20:
                break
            
            U_init = np.random.randn(num_u, rank) * 0.1
            V_init = np.random.randn(num_v, rank) * 0.1
            W_init = np.random.randn(num_w, rank) * 0.1
            
            # Scale to make initial loss reasonable
            # Each term should contribute roughly 1/(num_terms/rank) to each tensor entry
            scale = np.sqrt(num_u * num_v * num_w / (rank * n * m * p))
            U_init *= scale
            V_init *= scale
            W_init *= scale
            
            params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                from scipy.optimize import minimize
                result = minimize(
                    loss,
                    params0,
                    args=(rank,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10}
                )
                
                current_loss = result.fun
                if current_loss < best_loss:
                    n_u = num_u * rank
                    n_v = num_v * rank
                    U_new = result.x[:n_u].reshape((num_u, rank))
                    V_new = result.x[n_u:n_u + n_v].reshape((num_v, rank))
                    W_new = result.x[n_u + n_v:].reshape((num_w, rank))
                    
                    # Validate
                    if validate(U_new, V_new, W_new, rank):
                        best_U, best_V, best_W, best_rank = U_new, V_new, W_new, rank
                        best_loss = current_loss
                        # If we found a valid low-rank solution, we can stop
                        if best_rank <= 35:
                            break
            except Exception:
                pass
        
        if best_rank <= 35:
            break
    
    # If no improvement found, return schoolbook
    if best_rank == rank0:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
