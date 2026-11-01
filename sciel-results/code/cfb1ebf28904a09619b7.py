import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    n, m, p = DIMENSIONS
    
    # Construct target tensor M of shape (n*m, m*p, p*n)
    # Indices: a = i*m + j, b = j*p + k, c = k*n + i
    # M[a, b, c] = 1 if a,b,c correspond to i,j,k such that 
    # a = i*m+j, b = j*p+k, c = k*n+i.
    # Note: The mapping c = k*n + i is specific to the problem statement.
    
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    M_target = np.zeros((dim_u, dim_v, dim_w))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                M_target[a, b, c] = 1.0

    def verify(U, V, W, rank):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        # Reconstruct tensor
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, M_target)

    def objective(params, rank):
        # params: flattened U, V, W
        # U: (dim_u, rank), V: (dim_v, rank), W: (dim_w, rank)
        size_u = dim_u * rank
        size_v = dim_v * rank
        
        U_flat = params[:size_u]
        V_flat = params[size_u : size_u + size_v]
        W_flat = params[size_u + size_v :]
        
        U = U_flat.reshape(dim_u, rank)
        V = V_flat.reshape(dim_v, rank)
        W = W_flat.reshape(dim_w, rank)
        
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T_recon - M_target
        return np.sum(diff * diff)

    # Try to find rank 10 decomposition
    best_rank = 11
    best_U, best_V, best_W = None, None, None
    
    # Known rank 11 solution
    rank_known = 11
    U_known = [[0] * rank_known for _ in range(n * m)]
    V_known = [[0] * rank_known for _ in range(m * p)]
    W_known = [[0] * rank_known for _ in range(p * n)]

    terms_known = [
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(0, 1, 1), (1, 1, -1)]),
        ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)]),
        ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)]),
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)], [(0, 0, -1), (1, 0, 1)]),
        ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),
        ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),
        ([(0, 0, 1)], [(0, 2, 1)], [(2, 0, 1)]),
        ([(0, 1, 1)], [(1, 2, 1)], [(2, 0, 1)]),
        ([(1, 0, 1)], [(0, 2, 1)], [(2, 1, 1)]),
        ([(1, 1, 1)], [(1, 2, 1)], [(2, 1, 1)]),
    ]

    for r, (u_entries, v_entries, w_entries) in enumerate(terms_known):
        for i, j, val in u_entries:
            U_known[i * m + j][r] += val
        for j, k, val in v_entries:
            V_known[j * p + k][r] += val
        for k, i, val in w_entries:
            W_known[k * n + i][r] += val
            
    if verify(U_known, V_known, W_known, rank_known):
        best_U, best_V, best_W = U_known, V_known, W_known
        best_rank = rank_known

    # Search for rank 10
    if time.time() < deadline - 20: # Reserve time
        rank_to_try = 10
        num_restarts = 50
        for _ in range(num_restarts):
            if time.time() > deadline - 10:
                break
                
            # Random initialization
            # Scale matters. Let's try small integers.
            U_init = np.random.randn(dim_u, rank_to_try) * 2
            V_init = np.random.randn(dim_v, rank_to_try) * 2
            W_init = np.random.randn(dim_w, rank_to_try) * 2
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = minimize(objective, params_init, args=(rank_to_try,), method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-9})
                
                if res.success or res.fun < 1e-8:
                    # Extract U, V, W
                    size_u = dim_u * rank_to_try
                    size_v = dim_v * rank_to_try
                    U_flat = res.x[:size_u]
                    V_flat = res.x[size_u : size_u + size_v]
                    W_flat = res.x[size_u + size_v :]
                    
                    U = U_flat.reshape(dim_u, rank_to_try)
                    V = V_flat.reshape(dim_v, rank_to_try)
                    W = W_flat.reshape(dim_w, rank_to_try)
                    
                    # Check if close to integers/half-integers
                    # Multiply by 2 to check integers
                    U_int = np.round(U * 2) / 2.0
                    V_int = np.round(V * 2) / 2.0
                    W_int = np.round(W * 2) / 2.0
                    
                    if np.allclose(U, U_int) and np.allclose(V, V_int) and np.allclose(W, W_int):
                        if verify(U_int, V_int, W_int, rank_to_try):
                            best_rank = rank_to_try
                            best_U, best_V, best_W = U_int, V_int, W_int
                            break # Found optimal (if 10 is possible)
            except Exception:
                pass

    if best_U is None:
        # Fallback to standard 12
        best_rank = 12
        U_std = [[0]*12 for _ in range(4)]
        V_std = [[0]*12 for _ in range(6)]
        W_std = [[0]*12 for _ in range(6)]
        idx = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    a = i*2+j
                    b = j*3+k
                    c = k*2+i
                    U_std[a][idx] = 1
                    V_std[b][idx] = 1
                    W_std[c][idx] = 1
                    idx += 1
        best_U, best_V, best_W = U_std, V_std, W_std

    return {"U": best_U, "V": best_V, "W": best_W}
