import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS
    # T shape (n*m, m*p, p*n) = (9, 21, 21)
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
                
    best_rank = n * m * p
    best_U = np.zeros((n * m, best_rank))
    best_V = np.zeros((m * p, best_rank))
    best_W = np.zeros((p * n, best_rank))
    
    # Schoolbook
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j, col] = 1
                best_V[j * p + k, col] = 1
                best_W[k * n + i, col] = 1
                col += 1

    def solve_rank(r, num_trials=20):
        if time.time() > deadline:
            return None
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
                
            # Initialize
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Normalize
            for col_idx in range(r):
                norm = np.sqrt(np.sum(U[:, col_idx]**2) * np.sum(V[:, col_idx]**2) * np.sum(W[:, col_idx]**2))
                if norm > 1e-10:
                    U[:, col_idx] /= norm ** (1/3)
                    V[:, col_idx] /= norm ** (1/3)
                    W[:, col_idx] /= norm ** (1/3)
            
            # ALS
            for it in range(50):
                if time.time() > deadline:
                    return None
                    
                # Update U
                for a in range(n * m):
                    # M shape (m*p * p*n, r)
                    # vec(V[:,r] (x) W[:,r])
                    # We can construct M once outside the a-loop if V,W don't change?
                    # Yes, V and W are fixed during U update.
                    pass # Optimization: precompute M_VW, M_UW, M_UV outside loops?
                    
                # Precompute matrices for efficiency
                # M_VW: column r is vec(V[:,r] kron W[:,r])
                # Shape: (m*p * p*n, r) = (21*21, r) = (441, r)
                M_VW = np.zeros((m * p * p * n, r))
                for r_idx in range(r):
                    M_VW[:, r_idx] = np.kron(V[:, r_idx], W[:, r_idx])
                
                # Update U
                for a in range(n * m):
                    # T[a, :, :] is (m*p, p*n)
                    # vec(T[a, :, :]) is (m*p * p*n,)
                    target = T[a, :, :].flatten()
                    # Solve M_VW * U[a, :]^T = target
                    # U[a, :] is (r,)
                    # np.linalg.lstsq(M_VW, target)
                    sol, residuals, rank, sv = np.linalg.lstsq(M_VW, target, rcond=None)
                    U[a, :] = sol
                    
                # Precompute M_UW
                M_UW = np.zeros((n * m * p * n, r))
                for r_idx in range(r):
                    M_UW[:, r_idx] = np.kron(U[:, r_idx], W[:, r_idx])
                
                # Update V
                for b in range(m * p):
                    # Target is T[:, b, :] flattened?
                    # T[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
                    # For fixed b, T[:, b, :] is (n*m, p*n)
                    # We want sum_r V[b,r] (U[:,r] kron W[:,r])^T?
                    # vec(T[:, b, :]) = sum_r V[b,r] vec(U[:,r] (x) W[:,r])?
                    # No, U[:,r] is (n*m,), W[:,r] is (p*n,).
                    # The term is U[:,r] * V[b,r] * W[:,r]^T?
                    # T[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
                    # For fixed b, T[:, b, :] = sum_r V[b,r] (U[:,r] (x) W[:,r]^T)
                    # So vec(T[:, b, :]) = sum_r V[b,r] vec(U[:,r] (x) W[:,r]^T)
                    # My M_UW was constructed as kron(U, W).
                    # kron(U[:,r], W[:,r]) is vec(U[:,r] (x) W[:,r]^T) if we flatten row-major?
                    # np.kron(u, w) where u is (9,), w is (21,) gives (189,).
                    # u_i w_j.
                    # T[a, b, c] corresponds to index a * (p*n) + c in flatten?
                    # T shape (9, 21, 21).
                    # T[:, b, :] is (9, 21).
                    # Flatten is (189,).
                    # Index a * 21 + c.
                    # Term: U[a,r] * V[b,r] * W[c,r].
                    # Vector: sum_r V[b,r] * (U[:,r] (x) W[:,r])
                    # Yes, this matches.
                    
                    target = T[:, b, :].flatten()
                    sol, residuals, rank, sv = np.linalg.lstsq(M_UW, target, rcond=None)
                    V[b, :] = sol

                # Precompute M_UV
                M_UV = np.zeros((n * m * m * p, r))
                for r_idx in range(r):
                    M_UV[:, r_idx] = np.kron(U[:, r_idx], V[:, r_idx])
                
                # Update W
                for c in range(p * n):
                    # T[:, :, c] is (n*m, m*p)
                    # Flatten (189,)
                    # Term: sum_r W[c,r] (U[:,r] (x) V[:,r])
                    target = T[:, :, c].flatten()
                    sol, residuals, rank, sv = np.linalg.lstsq(M_UV, target, rcond=None)
                    W[c, :] = sol
                
                # Check error
                # Compute T_approx
                # T_approx[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
                # This is expensive to compute fully every iteration.
                # We can check a subset or compute error periodically.
                if it % 10 == 0:
                    # Compute error on a sample or full
                    # Full error:
                    # T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                    # err = np.linalg.norm(T - T_approx)
                    # If err < 1e-9, return
                    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                    err = np.linalg.norm(T - T_approx)
                    if err < 1e-9:
                        return U, V, W
                        
        return None

    # Try ranks from low to high
    # Start from a reasonable guess, e.g., 50
    for r in range(50, 63):
        if time.time() > deadline:
            break
        
        result = solve_rank(r, num_trials=10)
        if result is not None:
            best_rank = r
            best_U, best_V, best_W = result
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
