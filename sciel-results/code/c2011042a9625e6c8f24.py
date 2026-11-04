"""Find rank-minimal exact tensor decomposition for 2x4 by 4x7 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    start_time = time.time()
    deadline = 130  # Leave 20s margin
    
    # Build the target multiplication tensor
    T_tensor = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_tensor[a, b, c] = 1
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False
        rank = U.shape[1]
        T_approx = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T_approx += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
        return np.allclose(T_tensor, T_approx, atol=1e-6)
    
    def schoolbook():
        """Fallback: rank = n*m*p = 56."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, column] = 1
                    V[b, column] = 1
                    W[c, column] = 1
                    column += 1
        return U, V, W
    
    def als_decomposition(r, seed, max_iters=50):
        """Alternating Least Squares for tensor decomposition."""
        rng = np.random.default_rng(seed)
        U = rng.standard_normal((n*m, r)) * 0.1
        V = rng.standard_normal((m*p, r)) * 0.1
        W = rng.standard_normal((p*n, r)) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U: for each a, T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # Fix V, W. T[a] (flattened) = U[a,:] @ (V⊙W)^T
            B_u = np.zeros((m*p * p*n, r))
            for r_idx in range(r):
                B_u[:, r_idx] = np.kron(V[:, r_idx], W[:, r_idx])
            T_u = T_tensor.reshape(n*m, -1)
            try:
                U_new = T_u @ np.linalg.pinv(B_u.T)
            except Exception:
                return None
            
            # Update V
            B_v = np.zeros((n*m * p*n, r))
            for r_idx in range(r):
                B_v[:, r_idx] = np.kron(U[:, r_idx], W[:, r_idx])
            T_v = T_tensor.transpose(1, 0, 2).reshape(m*p, -1)
            try:
                V_new = T_v @ np.linalg.pinv(B_v.T)
            except Exception:
                return None
            
            # Update W
            B_w = np.zeros((n*m * m*p, r))
            for r_idx in range(r):
                B_w[:, r_idx] = np.kron(U[:, r_idx], V[:, r_idx])
            T_w = T_tensor.transpose(2, 0, 1).reshape(p*n, -1)
            try:
                W_new = T_w @ np.linalg.pinv(B_w.T)
            except Exception:
                return None
            
            U, V, W = U_new, V_new, W_new
        
        return U, V, W
    
    # Try to find a decomposition with rank lower than schoolbook
    # We know rank >= max(n*m, m*p, p*n) = max(8, 28, 14) = 28
    # Try from rank 28 up to 55
    
    best_result = None
    for rank in range(28, 56):
        if time.time() > deadline:
            break
        
        # Try multiple seeds for this rank
        for seed in [42, 123, 7, 99, 2023]:
            if time.time() > deadline:
                    break
            
            result = als_decomposition(rank, seed, max_iters=80)
            if result is not None:
                U, V, W = result
                if verify(U, V, W):
                    best_result = (rank, U, V, W)
                    break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        rank, U, V, W = best_result
        # Round to nearest half-integer if close
        def round_clean(M):
            M_r = np.round(M * 2) / 2
            if np.allclose(M, M_r, atol=1e-6):
                return M_r
            M_i = np.round(M)
            if np.allclose(M, M_i, atol=1e-6):
                return M_i
            return M
        U = round_clean(U)
        V = round_clean(V)
        W = round_clean(W)
        if verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
