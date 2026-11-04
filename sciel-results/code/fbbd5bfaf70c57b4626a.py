import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Dimensions of U, V, W
    dim_u = n * m  # 9
    dim_v = m * p  # 15
    dim_w = p * n  # 15
    
    # Construct target tensor T
    # T has shape (dim_u, dim_v, dim_w)
    # Index u = i*m + j
    # Index v = j*p + k
    # Index w = k*n + i
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    # Function to check if a decomposition is valid
    def is_valid(U, V, W, tol=1e-9):
        rank = U.shape[1]
        # Compute sum U_r \otimes V_r \otimes W_r
        # Reshape to (dim_u, dim_v, dim_w, rank)
        # Sum over rank
        # U: (dim_u, rank) -> (dim_u, 1, 1, rank)
        # V: (dim_v, rank) -> (1, dim_v, 1, rank)
        # W: (dim_w, rank) -> (1, 1, dim_w, rank)
        prod = U[:, np.newaxis, np.newaxis, :] * V[np.newaxis, :, np.newaxis, :] * W[np.newaxis, np.newaxis, :, :]
        # Sum over last axis
        result = np.sum(prod, axis=3)
        # Check difference
        diff = np.abs(result - T)
        return np.max(diff) < tol

    # Function to round to nearest integer or half-integer
    def round_to_half(X):
        # Multiply by 2, round to integer, divide by 2
        return np.round(X * 2) / 2.0

    # ALS Solver
    def solve_als(rank, num_iters=20, num_restarts=5, seed=0):
        rng = np.random.RandomState(seed)
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for restart in range(num_restarts):
            # Initialize
            U = rng.randn(dim_u, rank)
            V = rng.randn(dim_v, rank)
            W = rng.randn(dim_w, rank)
            
            # Normalize to avoid scale issues
            for r in range(rank):
                norm = np.linalg.norm(W[:, r])
                if norm > 1e-9:
                    W[:, r] /= norm
                    U[:, r] *= norm
            
            for it in range(num_iters):
                # Compute current sum
                # We can update one by one
                for r in range(rank):
                    # Remove current term
                    # We need the sum of all other terms
                    # Efficient way: maintain a running sum?
                    # For small rank, recomputing sum of others is O(R * dim_u * dim_v * dim_w)
                    # 42 * 9 * 15 * 15 = 85050 ops. Very fast.
                    
                    # Calculate sum of all terms except r
                    # prod_r = U[:, r, None, None] * V[None, :, r, None] * W[None, None, :, r]
                    # This is (dim_u, dim_v, dim_w)
                    
                    # To be efficient, we can precompute the full sum S
                    # S = sum_s U_s \otimes V_s \otimes W_s
                    # Then S_minus_r = S - U_r \otimes V_r \otimes W_r
                    # But S changes as we update.
                    # So we just compute S_minus_r directly.
                    
                    S_minus_r = np.zeros((dim_u, dim_v, dim_w))
                    for s in range(rank):
                        if s == r:
                            continue
                        S_minus_r += U[:, s, np.newaxis, np.newaxis] * V[np.newaxis, :, s, np.newaxis] * W[np.newaxis, np.newaxis, :, s]
                    
                    Residual = T - S_minus_r
                    
                    # Find best rank-1 approximation of Residual
                    # Unfold Residual to (dim_u, dim_v * dim_w)
                    Res_mat = Residual.reshape(dim_u, dim_v * dim_w)
                    # SVD
                    U_svd, s, Vt_svd = np.linalg.svd(Res_mat, full_matrices=False)
                    # Best rank-1 is U_svd[:, 0] * s[0] * Vt_svd[0, :]
                    # We need to distribute the scale between U, V, W
                    # Let u = U_svd[:, 0], v_w = Vt_svd[0, :]
                    # v_w is of length dim_v * dim_w
                    # We need to factor v_w into v (dim_v) and w (dim_w)
                    # This is a rank-1 factorization of the matrix V_w_mat = v_w.reshape(dim_v, dim_w)
                    V_w_mat = v_w.reshape(dim_v, dim_w)
                    U2, s2, Vt2 = np.linalg.svd(V_w_mat, full_matrices=False)
                    v = U2[:, 0]
                    w = Vt2[0, :]
                    
                    # Scale distribution
                    # We have u, v, w. We can scale them arbitrarily as long as product is constant.
                    # Let's normalize w to have norm 1, and distribute scale to u and v.
                    norm_w = np.linalg.norm(w)
                    if norm_w < 1e-9:
                        continue
                    w = w / norm_w
                    scale = s[0] * s2[0] * norm_w
                    # Distribute scale evenly or put in U?
                    # Let's put sqrt(scale) in U and V?
                    # Or just put scale in U.
                    U[:, r] = u * scale
                    V[:, r] = v
                    W[:, r] = w
                    
            # After iterations, round and check
            U_rounded = round_to_half(U)
            V_rounded = round_to_half(V)
            W_rounded = round_to_half(W)
            
            # Check validity
            # Compute result
            prod = U_rounded[:, np.newaxis, np.newaxis, :] * V_rounded[np.newaxis, :, np.newaxis, :] * W_rounded[np.newaxis, np.newaxis, :, :]
            result = np.sum(prod, axis=3)
            error = np.max(np.abs(result - T))
            
            if error < 1e-9:
                return U_rounded, V_rounded, W_rounded, error
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U_rounded.copy(), V_rounded.copy(), W_rounded.copy()
                
        return best_U, best_V, best_W, best_error

    # Try rank 42
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Try a few ranks starting from 42
    for rank in [42, 43, 44]:
        if time.time() > deadline:
            break
            
        # Try multiple seeds
        for seed in range(10):
            if time.time() > deadline:
                break
            U, V, W, err = solve_als(rank, num_iters=10, num_restarts=2, seed=seed)
            if err < 1e-9:
                # Convert to lists of lists for output
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
        
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

