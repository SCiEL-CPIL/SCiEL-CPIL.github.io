import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((dim_u, rank_school))
    V_school = np.zeros((dim_v, rank_school))
    W_school = np.zeros((dim_w, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    # Target tensor
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
                
    best_rank = rank_school
    best_decomp = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    start_time = time.time()
    deadline = 140  # 10s margin before 150s kill
    
    def verify_decomposition(U, V, W, tol=1e-9):
        T_recon = np.zeros((dim_u, dim_v, dim_w))
        for r in range(U.shape[1]):
            T_recon += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
        return np.allclose(T_recon, T, atol=tol)
    
    def als_optimize(U, V, W, max_iters=100, deadline=None):
        target_rank = U.shape[1]
        for iteration in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
                
            # Precompute M matrices for each factor update
            # Update U: T[i] = sum_r U[i,r] * outer(V[:,r], W[:,r])
            M_u = np.zeros((target_rank, dim_v * dim_w))
            for r in range(target_rank):
                M_u[r, :] = np.outer(V[:, r], W[:, r]).flatten()
            for i in range(dim_u):
                T_i_flat = T[i, :, :].flatten()
                try:
                    U[i, :], _, _, _ = np.linalg.lstsq(M_u.T, T_i_flat, rcond=None)
                except:
                    continue
                        
            # Update V: T[:,j,:] = sum_r V[j,r] * outer(U[:,r], W[:,r])
            M_v = np.zeros((target_rank, dim_u * dim_w))
            for r in range(target_rank):
                M_v[r, :] = np.outer(U[:, r], W[:, r]).flatten()
            for j in range(dim_v):
                T_j_flat = T[:, j, :].flatten()
                try:
                    V[j, :], _, _, _ = np.linalg.lstsq(M_v.T, T_j_flat, rcond=None)
                except:
                    continue
                        
            # Update W: T[:,:,k] = sum_r W[k,r] * outer(U[:,r], V[:,r])
            M_w = np.zeros((target_rank, dim_u * dim_v))
            for r in range(target_rank):
                M_w[r, :] = np.outer(U[:, r], V[:, r]).flatten()
            for k in range(dim_w):
                T_k_flat = T[:, :, k].flatten()
                try:
                    W[k, :], _, _, _ = np.linalg.lstsq(M_w.T, T_k_flat, rcond=None)
                except:
                    continue
                    
        return U, V, W
    
    # Try ranks from rank_school - 1 down to rank_school - 10
    for target_rank in range(rank_school - 1, rank_school - 11, -1):
        if time.time() > deadline:
            break
            
        # Multiple initialization strategies
        for attempt in range(30):
            if time.time() > deadline:
                break
                
            U = np.zeros((dim_u, target_rank))
            V = np.zeros((dim_v, target_rank))
            W = np.zeros((dim_w, target_rank))
            
            if attempt < 10:
                # Strategy 1: Dense random initialization
                scale = 0.1
                U = np.random.randn(dim_u, target_rank) * scale
                V = np.random.randn(dim_v, target_rank) * scale
                W = np.random.randn(dim_w, target_rank) * scale
            elif attempt < 20:
                # Strategy 2: Perturbed schoolbook subset
                indices = np.random.choice(rank_school, size=target_rank, replace=False)
                U = U_school[:, indices].copy()
                V = V_school[:, indices].copy()
                W = W_school[:, indices].copy()
                noise_scale = 0.01
                U += np.random.randn(dim_u, target_rank) * noise_scale
                V += np.random.randn(dim_v, target_rank) * noise_scale
                W += np.random.randn(dim_w, target_rank) * noise_scale
            else:
                # Strategy 3: Corner-peeling inspired structured init
                for r in range(target_rank):
                    if r < min(target_rank, dim_w):
                        k = r % p
                        i = r // p
                        if i < n:
                            j = np.random.randint(m)
                            U[i * m + j, r] = 1.0
                            V[j * p + k, r] = 1.0
                            W[k * n + i, r] = 1.0
                    else:
                        idx_u = np.random.randint(dim_u)
                        idx_v = np.random.randint(dim_v)
                        idx_w = np.random.randint(dim_w)
                        U[idx_u, r] = random.choice([0.5, 1, -0.5, -1])
                        V[idx_v, r] = random.choice([0.5, 1, -0.5, -1])
                        W[idx_w, r] = random.choice([0.5, 1, -0.5, -1])
            
            # Run ALS optimization
            U, V, W = als_optimize(U, V, W, max_iters=80, deadline=deadline)
            
            # Round to nearest half-integer
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify
            if verify_decomposition(U_rounded, V_rounded, W_rounded, tol=1e-9):
                best_rank = target_rank
                best_decomp = {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
                break
        
        if time.time() > deadline:
            break
    
    return best_decomp
