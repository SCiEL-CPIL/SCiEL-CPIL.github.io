import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 20
    
    start_time = time.time()
    deadline = start_time + 110
    
    # Precompute T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                l = k*n + i
                T[u, v, l] = 1.0
    
    T_flat = T.flatten()
    
    def get_rank_21():
        # ... same as before ...
        rank = 21
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        strassen = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
            ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
            ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
            ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
            ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
        ]
        
        r = 0
        for b in range(3):
            for t in range(7):
                u_coeffs, v_coeffs, w_coeffs = strassen[t]
                for idx, val in u_coeffs.items():
                    U[idx][r] = val
                for l_idx, val in v_coeffs.items():
                    if l_idx == 0: v_idx = 0 * p + 2 * b
                    elif l_idx == 1: v_idx = 0 * p + 2 * b + 1
                    elif l_idx == 2: v_idx = 1 * p + 2 * b
                    elif l_idx == 3: v_idx = 1 * p + 2 * b + 1
                    V[v_idx][r] = val
                for (i_idx, k_local), val in w_coeffs.items():
                    k_global = 2 * b + k_local
                    w_idx = k_global * n + i_idx
                    W[w_idx][r] = val
                r += 1
        return U, V, W

    U21, V21, W21 = get_rank_21()
    
    def try_rank_20():
        R = target_rank
        # Initial guess: first 20 columns of rank 21
        U_init = U21[:, :R].flatten()
        V_init = V21[:, :R].flatten()
        W_init = W21[:, :R].flatten()
        x0 = np.concatenate([U_init, V_init, W_init])
        
        size_u = n*m*R
        size_v = m*p*R
        
        def residual(x):
            U = x[:size_u].reshape(n*m, R)
            V = x[size_u:size_u+size_v].reshape(m*p, R)
            W = x[size_u+size_v:].reshape(p*n, R)
            
            # Compute tensor D
            # D[u, v, l] = sum_r U[u, r] V[v, r] W[l, r]
            D = np.einsum('ur,vr,lr->uvl', U, V, W)
            return D.flatten() - T_flat
            
        try:
            res = least_squares(residual, x0, method='lm', max_nfev=10000)
            if res.cost < 1e-8:
                # Check if we can round to half-integers
                U_opt = res.x[:size_u].reshape(n*m, R)
                V_opt = res.x[size_u:size_u+size_v].reshape(m*p, R)
                W_opt = res.x[size_u+size_v:].reshape(p*n, R)
                
                # Round to nearest 0.5
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                # Verify
                D_round = np.einsum('ur,vr,lr->uvl', U_round, V_round, W_round)
                if np.allclose(D_round.flatten(), T_flat, atol=1e-9):
                    return U_round, V_round, W_round
        except Exception:
            pass
        return None

    # Try rank 20
    if time.time() < deadline:
        result = try_rank_20()
        if result:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to Rank 21
    U, V, W = get_rank_21()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
