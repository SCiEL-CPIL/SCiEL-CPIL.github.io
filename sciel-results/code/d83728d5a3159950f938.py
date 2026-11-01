import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = 2, 3, 6
    dim_u = n * m  # 6
    dim_v = m * p  # 18
    dim_w = p * n  # 12
    
    # 1. Generate Schoolbook Decomposition (Rank 36)
    schoolbook_rank = n * m * p
    U_sb = np.zeros((dim_u, schoolbook_rank))
    V_sb = np.zeros((dim_v, schoolbook_rank))
    W_sb = np.zeros((dim_w, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    best_rank = schoolbook_rank
    
    # 2. Define the correct Matrix Multiplication Tensor for 2x3x6
    # T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i) else 0
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    # 3. Search for Rank < 36
    # We try rank 35 first. If found, it's an improvement.
    # Given the difficulty, we limit the search time.
    def find_decomp(rank, max_trials=20):
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                return None
            
            # Random initialization
            U = np.random.randn(dim_u, rank) * 0.5
            V = np.random.randn(dim_v, rank) * 0.5
            W = np.random.randn(dim_w, rank) * 0.5
            
            def params_to_tensors(params):
                U = params[:dim_u*rank].reshape(dim_u, rank)
                V = params[dim_u*rank:dim_u*rank + dim_v*rank].reshape(dim_v, rank)
                W = params[dim_u*rank + dim_v*rank:].reshape(dim_w, rank)
                return U, V, W
                
            def objective(params):
                U, V, W = params_to_tensors(params)
                # Vectorized computation of tensor approximation
                # Approx[u, v, w] = sum_r U[u,r] V[v,r] W[w,r]
                T_approx = np.einsum('ur,vr,wr->uvw', U, V, W)
                diff = T - T_approx
                return np.sum(diff**2)
                
            initial_params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                # Use L-BFGS-B for local optimization
                result = minimize(objective, initial_params, method='L-BFGS-B', maxiter=100, 
                                  options={'maxls': 20, 'gtol': 1e-5})
                if result.fun < 1e-8:
                    U, V, W = params_to_tensors(result.x)
                    # Round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    # Verify exactness
                    T_approx = np.einsum('ur,vr,wr->uvw', U_r, V_r, W_r)
                    if np.allclose(T_approx, T, atol=1e-6):
                        return U_r.tolist(), V_r.tolist(), W_r.tolist()
            except Exception:
                pass
        return None

    # Try rank 35
    # Note: Rank 35 is unlikely to be found quickly, but we try.
    decomp = find_decomp(35, max_trials=10)
    if decomp:
        best_U, best_V, best_W = decomp
        best_rank = 35

    return {"U": best_U, "V": best_V, "W": best_W}
