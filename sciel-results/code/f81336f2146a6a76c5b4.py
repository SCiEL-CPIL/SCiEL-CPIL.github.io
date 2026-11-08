import numpy as np
import time

def construct():
    start = time.time()
    deadline = start + 110.0

    n, m, p = 2, 3, 4
    dim_u, dim_v, dim_w = n * m, m * p, p * n  # 6, 12, 8

    # Build the multiplication tensor
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for i in range(dim_u):
        ai, aj = i // m, i % m
        for j in range(dim_v):
            bj, bj2 = j // p, j % p
            if aj == bj:
                for k in range(dim_w):
                    ck, ck2 = k % n, k // n
                    if ai == ck and bj2 == ck2:
                        T[i, j, k] = 1.0

    # Try to find rank 20 using alternating minimization with multiple restarts
    def try_rank(R, n_restarts=5, max_iter=300, seed=0):
        if time.time() > deadline - 30:
            return None
        
        best = None
        best_err = np.inf
        
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            
            rng = np.random.RandomState(seed + restart)
            
            # Initialize with small random values
            U = rng.randn(dim_u, R) * 0.1
            V = rng.randn(dim_v, R) * 0.1
            W = rng.randn(dim_w, R) * 0.1
            
            for it in range(max_iter):
                if time.time() > deadline - 10:
                    break
                
                # Update W
                for r in range(R):
                    R_r = T.copy()
                    for r2 in range(R):
                        if r2 != r:
                            R_r -= np.einsum('i,j,k->ijk', U[:, r2], V[:, r2], W[:, r2])
                    
                    den = np.sum(U[:, r] ** 2) * np.sum(V[:, r] ** 2)
                    if abs(den) < 1e-15:
                        continue
                    num = np.array([np.sum(R_r[:, :, k] * np.outer(U[:, r], V[:, r])) for k in range(dim_w)])
                    W[:, r] = num / den
                
                # Update V
                for r in range(R):
                    R_r = T.copy()
                    for r2 in range(R):
                        if r2 != r:
                            R_r -= np.einsum('i,j,k->ijk', U[:, r2], V[:, r2], W[:, r2])
                    
                    den = np.sum(U[:, r] ** 2) * np.sum(W[:, r] ** 2)
                    if abs(den) < 1e-15:
                        continue
                    num = np.array([np.sum(R_r[:, j, :] * np.outer(U[:, r], W[:, r])) for j in range(dim_v)])
                    V[:, r] = num / den
                
                # Update U
                for r in range(R):
                    R_r = T.copy()
                    for r2 in range(R):
                        if r2 != r:
                            R_r -= np.einsum('i,j,k->ijk', U[:, r2], V[:, r2], W[:, r2])
                    
                    den = np.sum(V[:, r] ** 2) * np.sum(W[:, r] ** 2)
                    if abs(den) < 1e-15:
                        continue
                    num = np.array([np.sum(R_r[i, :, :] * np.outer(V[:, r], W[:, r])) for i in range(dim_u)])
                    U[:, r] = num / den
                
                # Compute error
                T_approx = np.zeros_like(T)
                for r in range(R):
                    T_approx += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                err = np.linalg.norm(T - T_approx)
                
                if err < best_err:
                    best_err = err
                    best = (U.copy(), V.copy(), W.copy())
                
                if best_err < 1e-10:
                    break
        
        if best is not None and best_err < 1e-8:
            return best
        return None

    # Try rank 20
    result = try_rank(20, n_restarts=3, max_iter=200)
    if result is not None:
        U, V, W = result
        T_approx = np.zeros_like(T)
        for r in range(20):
            T_approx += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        if np.allclose(T, T_approx, atol=1e-6):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: rank-22 construction
    rank = 22
    U = np.zeros((dim_u, rank), dtype=float)
    V = np.zeros((dim_v, rank), dtype=float)
    W = np.zeros((dim_w, rank), dtype=float)
    
    def add_term(col_idx, u_vals, v_vals, w_vals):
        for idx, val in u_vals.items():
            U[idx, col_idx] += val
        for idx, val in v_vals.items():
            V[idx, col_idx] += val
        for idx, val in w_vals.items():
            W[idx, col_idx] += val
    
    add_term(0, {0: 1, 4: 1}, {0: 1, 5: 1}, {0: 1, 3: 1})
    add_term(1, {3: 1, 4: 1}, {0: 1}, {1: 1, 3: -1})
    add_term(2, {0: 1}, {1: 1, 5: -1}, {2: 1, 3: 1})
    add_term(3, {4: 1}, {4: 1, 0: -1}, {0: 1, 1: 1})
    add_term(4, {0: 1, 1: 1}, {5: 1}, {0: -1, 2: 1})
    add_term(5, {3: 1, 0: -1}, {0: 1, 1: 1}, {3: 1})
    add_term(6, {1: 1, 4: -1}, {4: 1, 5: 1}, {0: 1})
    
    add_term(7, {0: 1, 4: 1}, {2: 1, 7: 1}, {4: 1, 7: 1})
    add_term(8, {3: 1, 4: 1}, {2: 1}, {5: 1, 7: -1})
    add_term(9, {0: 1}, {3: 1, 7: -1}, {6: 1, 7: 1})
    add_term(10, {4: 1}, {6: 1, 2: -1}, {4: 1, 5: 1})
    add_term(11, {0: 1, 1: 1}, {7: 1}, {4: -1, 6: 1})
    add_term(12, {3: 1, 0: -1}, {2: 1, 3: 1}, {7: 1})
    add_term(13, {1: 1, 4: -1}, {6: 1, 7: 1}, {4: 1})
    
    add_term(14, {2: 1}, {8: 1}, {0: 1})
    add_term(15, {2: 1}, {9: 1}, {2: 1})
    add_term(16, {2: 1}, {10: 1}, {4: 1})
    add_term(17, {2: 1}, {11: 1}, {6: 1})
    add_term(18, {5: 1}, {8: 1}, {1: 1})
    add_term(19, {5: 1}, {9: 1}, {3: 1})
    add_term(20, {5: 1}, {10: 1}, {5: 1})
    add_term(21, {5: 1}, {11: 1}, {7: 1})
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
