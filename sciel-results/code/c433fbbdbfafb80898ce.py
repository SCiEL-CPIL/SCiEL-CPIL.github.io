import numpy as np
import time

DIMENSIONS = [2, 2, 5]

def construct():
    start_time = time.time()
    deadline = 140.0
    
    n, m, p = DIMENSIONS
    
    # Precompute T
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    # Precompute index grids
    b_idx = np.arange(m * p)
    c_idx = np.arange(p * n)
    b_grid, c_grid = np.meshgrid(b_idx, c_idx, indexing='ij')
    b_flat = b_grid.ravel()
    c_flat = c_grid.ravel()
    
    a_idx = np.arange(n * m)
    c_idx = np.arange(p * n)
    a_grid, c_grid = np.meshgrid(a_idx, c_idx, indexing='ij')
    a_flat = a_grid.ravel()
    c_flat = c_grid.ravel()
    
    a_idx = np.arange(n * m)
    b_idx = np.arange(m * p)
    a_grid, b_grid = np.meshgrid(a_idx, b_idx, indexing='ij')
    a_flat = a_grid.ravel()
    b_flat = b_grid.ravel()
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    for rank in range(14, 17):
        if time.time() > deadline:
            break
            
        for restart in range(20):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + restart + rank * 100)
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            for it in range(200):
                if time.time() > deadline:
                    break
                    
                # Update U
                for a in range(n * m):
                    A_a = V[b_flat, :] * W[c_flat, :]
                    b_a = T[a].ravel()
                    U[a, :], residuals, rank_u, sv = np.linalg.lstsq(A_a, b_a, rcond=None)
                    
                # Update V
                for b in range(m * p):
                    A_b = U[a_flat, :] * W[c_flat, :]
                    c_b = T[:, b, :].ravel()
                    V[b, :], residuals, rank_v, sv = np.linalg.lstsq(A_b, c_b, rcond=None)
                    
                # Update W
                for c in range(p * n):
                    A_c = U[a_flat, :] * V[b_flat, :]
                    d_c = T[:, :, c].ravel()
                    W[c, :], residuals, rank_w, sv = np.linalg.lstsq(A_c, d_c, rcond=None)
                    
                # Check error
                # Vectorized error check
                # U[a, :] dot (V[b, :] * W[c, :])
                # This is tricky to vectorize fully for all a,b,c at once without large intermediate arrays.
                # But n*m=4, m*p=10, p*n=10.
                # We can loop over a, and vectorize b,c.
                
                max_err = 0.0
                for a in range(n * m):
                    # Val[b, c] = sum_r U[a, r] V[b, r] W[c, r]
                    # = (U[a, :] * V.T) @ W.T ?
                    # Let M = U[a, :] * V.T -> (rank, m*p)
                    # M @ W.T -> (m*p, p*n)
                    M = U[a, :] @ V.T # (rank, m*p) -> (m*p, rank)? No.
                    # U[a, :] is (rank,). V is (m*p, rank).
                    # V.T is (rank, m*p).
                    # U[a, :] @ V.T is (m*p,).
                    # Let's call this vec.
                    vec = U[a, :] @ V.T # (m*p,)
                    # Now Val[b, c] = sum_r vec[b] W[c, r]? No.
                    # Val[b, c] = sum_r U[a, r] V[b, r] W[c, r]
                    # = sum_r (U[a, r] V[b, r]) W[c, r]
                    # Let Y[b, r] = U[a, r] V[b, r].
                    # Y = np.outer(U[a, :], V) ? No.
                    # Y[b, r] = V[b, r] * U[a, r].
                    # Y = V * U[a, :] (broadcasting). Shape (m*p, rank).
                    # Then Val = Y @ W.T. Shape (m*p, p*n).
                    Y = V * U[a, :]
                    Val = Y @ W.T
                    Target = T[a]
                    err = np.max(np.abs(Val - Target))
                    if err > max_err:
                        max_err = err
                        
                if max_err < 1e-10:
                    # Snap
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    # Verify
                    valid = True
                    for a in range(n * m):
                        Y = V_snap * U_snap[a, :]
                        Val = Y @ W_snap.T
                        if np.max(np.abs(Val - T[a])) > 1e-9:
                            valid = False
                            break
                            
                    if valid:
                        return {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }
                        
                    # Try integer
                    U_int = np.round(U)
                    V_int = np.round(V)
                    W_int = np.round(W)
                    
                    valid = True
                    for a in range(n * m):
                        Y = V_int * U_int[a, :]
                        Val = Y @ W_int.T
                        if np.max(np.abs(Val - T[a])) > 1e-9:
                            valid = False
                            break
                            
                    if valid:
                        return {
                            "U": U_int.tolist(),
                            "V": V_int.tolist(),
                            "W": W_int.tolist()
                        }
                        
    return schoolbook()
