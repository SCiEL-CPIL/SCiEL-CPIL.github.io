import time
import numpy as np
from itertools import product

def construct():
    n, m, p = 3, 3, 4
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Build target tensor T[a,b,c]
    # a = i*m + j (i=0..n-1, j=0..m-1), b = j*p + k (j=0..m-1, k=0..p-1), c = k*n + i (k=0..p-1, i=0..n-1)
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook fallback
    rank_fb = n * m * p
    U_fb = np.zeros((n*m, rank_fb), dtype=float)
    V_fb = np.zeros((m*p, rank_fb), dtype=float)
    W_fb = np.zeros((p*n, rank_fb), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1

    best_result = (U_fb, V_fb, W_fb)
    best_rank = rank_fb

    def compute_decomp(U, V, W):
        return np.einsum('at,bt,ct->abc', U, V, W, optimize=True)

    def check_decomp(U, V, W, atol=1e-9):
        T_decomp = compute_decomp(U, V, W)
        return np.allclose(T_decomp, T, atol=atol)

    def snap_to_half(U, V, W):
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        return U_s, V_s, W_s

    def snap_to_quarter(U, V, W):
        U_s = np.round(U * 4) / 4
        V_s = np.round(V * 4) / 4
        W_s = np.round(W * 4) / 4
        return U_s, V_s, W_s

    def alt_least_squares(U, V, W, rank, max_iters=80, tol=1e-10):
        """Alternating least squares to refine the decomposition."""
        size_u, size_v, size_w = n*m, m*p, p*n
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Update U
            for a_idx in range(size_u):
                A = np.einsum('bt,ct->bct', V, W).reshape(size_v * size_w, rank)
                b_vec = T[a_idx, :, :].reshape(-1)
                try:
                    U[a_idx, :] = np.linalg.lstsq(A.T, b_vec, rcond=None)[0]
                except:
                    pass
            
            # Update V
            for b_idx in range(size_v):
                A = np.einsum('at,ct->act', U, W).reshape(size_u * size_w, rank)
                b_vec = T[:, b_idx, :].reshape(-1)
                try:
                    V[b_idx, :] = np.linalg.lstsq(A.T, b_vec, rcond=None)[0]
                except:
                    pass
            
            # Update W
            for c_idx in range(size_w):
                A = np.einsum('at,bt->abt', U, V).reshape(size_u * size_v, rank)
                b_vec = T[:, :, c_idx].reshape(-1)
                try:
                    W[c_idx, :] = np.linalg.lstsq(A.T, b_vec, rcond=None)[0]
                except:
                    pass
            
            # Check convergence
            T_decomp = compute_decomp(U, V, W)
            err = np.sum((T - T_decomp) ** 2)
            if err < tol:
                break
        
        return U, V, W

    def try_rank_als(rank, num_trials=20):
        """Try to find a rank-r decomposition using ALS with various initializations."""
        size_u, size_v, size_w = n*m, m*p, p*n
        
        for trial in range(num_trials):
            if time.time() > deadline - 15:
                return None
            
            # Multiple initialization strategies
            if trial % 5 == 0:
                # Small random
                np.random.seed(trial * 1000 + rank)
                U = np.random.randn(size_u, rank) * 0.3
                V = np.random.randn(size_v, rank) * 0.3
                W = np.random.randn(size_w, rank) * 0.3
            elif trial % 5 == 1:
                # Medium random
                np.random.seed(trial * 1000 + rank + 1)
                U = np.random.randn(size_u, rank) * 0.7
                V = np.random.randn(size_v, rank) * 0.7
                W = np.random.randn(size_w, rank) * 0.7
            elif trial % 5 == 2:
                # Sparse random
                np.random.seed(trial * 1000 + rank + 2)
                U = np.random.randn(size_u, rank) * 1.5
                V = np.random.randn(size_v, rank) * 1.5
                W = np.random.randn(size_w, rank) * 1.5
                U[np.abs(U) < 0.8] = 0.0
                V[np.abs(V) < 0.8] = 0.0
                W[np.abs(W) < 0.8] = 0.0
            elif trial % 5 == 3:
                # Binary-ish random
                np.random.seed(trial * 1000 + rank + 3)
                U = np.random.choice([-1, 0, 1], size=(size_u, rank)) * 0.5
                V = np.random.choice([-1, 0, 1], size=(size_v, rank)) * 0.5
                W = np.random.choice([-1, 0, 1], size=(size_w, rank)) * 0.5
            else:
                # Structured: initialize with schoolbook-like patterns
                np.random.seed(trial * 1000 + rank + 4)
                U = np.zeros((size_u, rank))
                V = np.zeros((size_v, rank))
                W = np.zeros((size_w, rank))
                # Randomly assign some 1s
                for t in range(rank):
                    a_idx = np.random.randint(0, size_u)
                    b_idx = np.random.randint(0, size_v)
                    c_idx = np.random.randint(0, size_w)
                    U[a_idx, t] = 1.0
                    V[b_idx, t] = 1.0
                    W[c_idx, t] = 1.0
                    # Add some noise
                    U[np.random.randint(0, size_u), t] += np.random.randn() * 0.3
                    V[np.random.randint(0, size_v), t] += np.random.randn() * 0.3
                    W[np.random.randint(0, size_w), t] += np.random.randn() * 0.3
            
            # Run ALS
            U, V, W = alt_least_squares(U, V, W, rank, max_iters=60, tol=1e-10)
            
            # Check
            if check_decomp(U, V, W):
                U_s, V_s, W_s = snap_to_half(U, V, W)
                if check_decomp(U_s, V_s, W_s):
                    return U_s, V_s, W_s
                else:
                    U_s2, V_s2, W_s2 = snap_to_quarter(U, V, W)
                    if check_decomp(U_s2, V_s2, W_s2):
                        return U_s2, V_s2, W_s2
                    else:
                        return U, V, W
        
        return None

    def try_rank_optimize(rank, num_trials=8):
        """Try to find a rank-r decomposition using scipy optimization."""
        from scipy.optimize import minimize
        
        size_u, size_v, size_w = n*m, m*p, p*n
        total_vars = (size_u + size_v + size_w) * rank
        
        def objective(x):
            U = x[:size_u*rank].reshape(size_u, rank)
            V = x[size_u*rank:size_u*rank + size_v*rank].reshape(size_v, rank)
            W = x[size_u*rank + size_v*rank:].reshape(size_w, rank)
            T_decomp = compute_decomp(U, V, W)
            return np.sum((T - T_decomp) ** 2)
        
        for trial in range(num_trials):
            if time.time() > deadline - 15:
                return None
            
            if trial % 3 == 0:
                np.random.seed(trial * 1000 + rank)
                x0 = np.random.randn(total_vars) * 0.5
            elif trial % 3 == 1:
                np.random.seed(trial * 1000 + rank + 1)
                x0 = np.random.randn(total_vars) * 1.0
            else:
                np.random.seed(trial * 1000 + rank + 2)
                x0 = np.random.randn(total_vars) * 2.0
                x0[np.abs(x0) < 1.0] = 0.0
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', 
                               options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.fun < 1e-8:
                    x_opt = res.x
                    U = x_opt[:size_u*rank].reshape(size_u, rank)
                    V = x_opt[size_u*rank:size_u*rank + size_v*rank].reshape(size_v, rank)
                    W = x_opt[size_u*rank + size_v*rank:].reshape(size_w, rank)
                    
                    U_s, V_s, W_s = snap_to_half(U, V, W)
                    if check_decomp(U_s, V_s, W_s):
                        return U_s, V_s, W_s
                    else:
                        if check_decomp(U, V, W):
                            U_s2, V_s2, W_s2 = snap_to_quarter(U, V, W)
                            if check_decomp(U_s2, V_s2, W_s2):
                                return U_s2, V_s2, W_s2
                            else:
                                return U, V, W
            except Exception:
                pass
        
        return None

    # Try ranks from low to high
    for rank in range(26, 37):
        if time.time() > deadline - 20:
            break
        if rank >= best_rank:
            continue
        
        # First try ALS approach
        result = try_rank_als(rank, num_trials=25)
        if result is None:
            # Then try generic optimization
            result = try_rank_optimize(rank, num_trials=10)
        
        if result is not None:
            U_r, V_r, W_r = result
            best_result = (U_r, V_r, W_r)
            best_rank = rank
            if rank == 26:
                break

    U_best, V_best, W_best = best_result
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
