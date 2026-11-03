"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_schoolbook = [[0] * rank_schoolbook for _ in range(n * m)]
    V_schoolbook = [[0] * rank_schoolbook for _ in range(m * p)]
    W_schoolbook = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j][col] = 1
                V_schoolbook[j * p + k][col] = 1
                W_schoolbook[k * n + i][col] = 1
                col += 1
    
    # Target Tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    start_time = time.time()
    time_limit = 110.0 # Safety margin for 120s limit

    best_U = U_schoolbook
    best_V = V_schoolbook
    best_W = W_schoolbook
    best_rank = rank_schoolbook

    # Try ranks from 59 down to 40
    for r in range(59, 39, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try multiple restarts for this rank
        # Greedy initialization + Power method polish
        num_restarts = 50
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize U, V, W
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            
            T_curr = T.copy()
            
            # Greedy rank-1 approximation
            for c in range(r):
                if time.time() - start_time > time_limit:
                    break
                    
                # Find best rank-1 approx to T_curr
                # Initialize with random vectors
                u = np.random.randn(n*m)
                v = np.random.randn(m*p)
                w = np.random.randn(p*n)
                
                # Power method
                for _ in range(20):
                    u_new = np.einsum('ijk,j,k->i', T_curr, v, w)
                    v_new = np.einsum('ijk,i,k->j', T_curr, u, w)
                    w_new = np.einsum('ijk,i,j->k', T_curr, u, v)
                    
                    norm_u = np.linalg.norm(u_new)
                    norm_v = np.linalg.norm(v_new)
                    norm_w = np.linalg.norm(w_new)
                    
                    if norm_u == 0 or norm_v == 0 or norm_w == 0:
                        break
                    
                    u_new /= norm_u
                    v_new /= norm_v
                    w_new /= norm_w
                    
                    u, v, w = u_new, v_new, w_new
                
                # Scale to match magnitude
                uo = np.einsum('i,j,k->ijk', u, v, w)
                denom = np.sum(uo * uo)
                if denom == 0:
                    continue
                s = np.sum(T_curr * uo) / denom
                
                U[:, c] = s * u
                V[:, c] = s * v
                W[:, c] = s * w
                
                # Update T_curr
                T_curr = T_curr - np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
            
            # Check if residual is small
            residual = T - np.einsum('ic,jc,kc->ijk', U, V, W)
            if np.linalg.norm(residual) < 1e-9:
                # Try to round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify
                check_sum = np.einsum('ic,jc,kc->ijk', U_rounded, V_rounded, W_rounded)
                if np.allclose(check_sum, T, atol=1e-9):
                    best_U = U_rounded.tolist()
                    best_V = V_rounded.tolist()
                    best_W = W_rounded.tolist()
                    best_rank = r
                    break # Found a valid rank r, try lower rank
        
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
