import time
import numpy as np
from itertools import product
from scipy.optimize import minimize

def construct():
    n, m, p = 3, 3, 4
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # Build target tensor T[a,b,c]
    # a = i*m + j, b = j*p + k, c = k*n + i
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

    def check_decomp(U, V, W):
        T_decomp = compute_decomp(U, V, W)
        return np.allclose(T_decomp, T, atol=1e-9)

    def snap_to_half(U, V, W):
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        return U_s, V_s, W_s

    def alt_least_squares(U, V, W, rank, max_iters=50, tol=1e-8):
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

    def objective(x, rank):
        size_u, size_v, size_w = n*m, m*p, p*n
        U = x[:size_u*rank].reshape(size_u, rank)
        V = x[size_u*rank:size_u*rank + size_v*rank].reshape(size_v, rank)
        W = x[size_u*rank + size_v*rank:].reshape(size_w, rank)
        
        T_decomp = compute_decomp(U, V, W)
        return np.sum((T - T_decomp) ** 2)

    def try_rank(rank, num_trials=10):
        """Try to find a rank-r decomposition using scipy optimization."""
        size_u, size_v, size_w = n*m, m*p, p*n
        total_vars = (size_u + size_v + size_w) * rank
        
        for trial in range(num_trials):
            if time.time() > deadline - 15:
                return None
            
            # Multiple initialization strategies
            if trial % 3 == 0:
                # Small random
                np.random.seed(trial * 1000 + rank)
                x0 = np.random.randn(total_vars) * 0.5
            elif trial % 3 == 1:
                # Larger random
                np.random.seed(trial * 1000 + rank + 1)
                x0 = np.random.randn(total_vars) * 1.0
            else:
                # Sparse random (many zeros)
                np.random.seed(trial * 1000 + rank + 2)
                x0 = np.random.randn(total_vars) * 2.0
                x0[np.abs(x0) < 1.0] = 0.0
            
            # Run L-BFGS-B optimization
            try:
                res = minimize(objective, x0, args=(rank,), method='L-BFGS-B', 
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.fun < 1e-8:
                    x_opt = res.x
                    U = x_opt[:size_u*rank].reshape(size_u, rank)
                    V = x_opt[size_u*rank:size_u*rank + size_v*rank].reshape(size_v, rank)
                    W = x_opt[size_u*rank + size_v*rank:].reshape(size_w, rank)
                    
                    # Try to snap to half-integers
                    U_s, V_s, W_s = snap_to_half(U, V, W)
                    if check_decomp(U_s, V_s, W_s):
                        return U_s, V_s, W_s
                    else:
                        # Check if the unsnapped version is effectively exact
                        if check_decomp(U, V, W):
                            # Try more aggressive snapping
                            U_s2 = np.round(U * 4) / 4
                            V_s2 = np.round(V * 4) / 4
                            W_s2 = np.round(W * 4) / 4
                            if check_decomp(U_s2, V_s2, W_s2):
                                return U_s2, V_s2, W_s2
                            # Return the float version - might still be accepted
                            return U, V, W
            except Exception:
                pass
        
        return None

    def try_structured_split(rank):
        """
        Try splitting approach: decompose 3x3x4 into structured parts.
        Strategy: Use the fact that we can view this as:
        - Multiply 3x3 by 3x2 (left half) and 3x3 by 3x2 (right half)
        - Or use block structure
        
        For 3x3x2, we can try rank-13 or lower decompositions.
        Then combine two 3x3x2 decompositions for the full 3x3x4.
        """
        size_u, size_v, size_w = n*m, m*p, p*n
        
        # Try to find a decomposition where the W matrix has structure
        # that reflects the 4 columns being split into two groups of 2
        
        # Approach: parameterize W to have block structure
        # W has shape (p*n, rank) = (12, rank)
        # c = k*n + i, so c ranges over k=0..3, i=0..2
        # Group c by k: c in {0,1,2} (k=0), {3,4,5} (k=1), {6,7,8} (k=2), {9,10,11} (k=3)
        
        # Try optimization with structured initialization
        for trial in range(5):
            if time.time() > deadline - 15:
                return None
            
            np.random.seed(10000 + trial * 100 + rank)
            
            # Initialize with some structure
            U = np.random.randn(size_u, rank) * 0.5
            V = np.random.randn(size_v, rank) * 0.5
            W = np.random.randn(size_w, rank) * 0.5
            
            # Run ALS
            U, V, W = alt_least_squares(U, V, W, rank, max_iters=30, tol=1e-10)
            
            # Check
            if check_decomp(U, V, W):
                U_s, V_s, W_s = snap_to_half(U, V, W)
                if check_decomp(U_s, V_s, W_s):
                    return U_s, V_s, W_s
                else:
                    return U, V, W
        
        return None

    # Try ranks from 26 to 35
    # Known theoretical bound is 26.
    for rank in range(26, 36):
        if time.time() > deadline - 20:
            break
        if rank >= best_rank:
            continue
        
        # First try structured approach
        result = try_structured_split(rank)
        if result is None:
            # Then try generic optimization
            result = try_rank(rank, num_trials=12)
        
        if result is not None:
            U_r, V_r, W_r = result
            best_result = (U_r, V_r, W_r)
            best_rank = rank
            # If we found the theoretical minimum, we can stop
            if rank == 26:
                break

    U_best, V_best, W_best = best_result
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
