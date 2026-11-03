import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    size_u, size_v, size_w = n * m, m * p, p * n
    
    # Build target tensor T[a,b,c]
    T = np.zeros((size_u, size_v, size_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook fallback (rank 36)
    rank_fb = size_u * size_v * size_w // (n * m * p)  # This is wrong, let's compute properly
    rank_fb = n * m * p
    U_fb = np.zeros((size_u, rank_fb), dtype=np.float64)
    V_fb = np.zeros((size_v, rank_fb), dtype=np.float64)
    W_fb = np.zeros((size_w, rank_fb), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1.0
                V_fb[j * p + k, col] = 1.0
                W_fb[k * n + i, col] = 1.0
                col += 1
    
    best_result = (U_fb, V_fb, W_fb)
    best_rank = rank_fb
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    def compute_decomp(U, V, W):
        return np.einsum('at,bt,ct->abc', U, V, W, optimize=True)
    
    def check_exact(U, V, W):
        """Check if decomposition is exact (all entries are 0 or 1 in the right places)."""
        T_decomp = compute_decomp(U, V, W)
        # Check that T_decomp matches T exactly
        return np.allclose(T_decomp, T, atol=1e-10)
    
    def is_half_integer(arr):
        """Check if all entries are integers or half-integers."""
        doubled = arr * 2
        rounded = np.round(doubled)
        return np.allclose(doubled, rounded, atol=1e-8)
    
    def snap_to_half(U, V, W):
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        return U_s, V_s, W_s
    
    def alt_least_squares(U, V, W, rank, max_iters=20, tol=1e-10):
        """Alternating least squares to refine the decomposition."""
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Update U
            for a in range(size_u):
                A = np.einsum('bt,ct->bct', V, W).reshape(size_v * size_w, rank)
                b_vec = T[a, :, :].reshape(-1)
                try:
                    U[a, :] = np.linalg.lstsq(A.T, b_vec, rcond=None)[0]
                except:
                    pass
            
            # Update V
            for b in range(size_v):
                A = np.einsum('at,ct->act', U, W).reshape(size_u * size_w, rank)
                b_vec = T[:, b, :].reshape(-1)
                try:
                    V[b, :] = np.linalg.lstsq(A.T, b_vec, rcond=None)[0]
                except:
                    pass
            
            # Update W
            for c in range(size_w):
                A = np.einsum('at,bt->abt', U, V).reshape(size_u * size_v, rank)
                b_vec = T[:, :, c].reshape(-1)
                try:
                    W[c, :] = np.linalg.lstsq(A.T, b_vec, rcond=None)[0]
                except:
                    pass
            
            # Check convergence
            T_decomp = compute_decomp(U, V, W)
            err = np.sum((T - T_decomp) ** 2)
            if err < tol:
                break
        
        return U, V, W
    
    # Try ranks from just below 36 down to a reasonable lower bound
    # Theoretical lower bound for 3x3x4 is at least max(9, 12, 12) = 12, but realistically higher
    # Try from 35 down to 25 (giving up if we can't find better)
    for rank in range(rank_fb - 1, 24, -1):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        num_trials = 5
        for trial in range(num_trials):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(rank * 1000 + trial)
            U = np.random.randn(size_u, rank) * 0.5
            V = np.random.randn(size_v, rank) * 0.5
            W = np.random.randn(size_w, rank) * 0.5
            
            # Run ALS
            U, V, W = alt_least_squares(U, V, W, rank, max_iters=15)
            
            # Check if we got a good solution
            if check_exact(U, V, W):
                # Try to snap to half-integers
                U_s, V_s, W_s = snap_to_half(U, V, W)
                if check_exact(U_s, V_s, W_s) and is_half_integer(U_s) and is_half_integer(V_s) and is_half_integer(W_s):
                    return {
                        "U": U_s.tolist(),
                        "V": V_s.tolist(),
                        "W": W_s.tolist()
                    }
                else:
                    # If unsnapped is exact but not half-integer, we can't use it
                    # But if it's close to half-integers, maybe we can use it?
                    # The problem requires integer/half-integer entries, so we must snap
                    pass
    
    # If no better rank found, return schoolbook
    return {
        "U": U_fb.tolist(),
        "V": V_fb.tolist(),
        "W": W_fb.tolist()
    }
