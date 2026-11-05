import numpy as np
import time

def construct():
    n, m, p = 3, 3, 4
    t0 = time.time()
    deadline = t0 + 115

    # Dimensions
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n

    # Target Tensor T
    # T is a 3D array of shape (dim_U, dim_V, dim_W)
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(i,k)
    # We can store it as a list of non-zero entries to save memory and time
    # But for ALS, we need to form the matrix M.
    # M has rows indexed by (a, b) and columns by r.
    # M[(a,b), r] = U[a, r] * V[b, r]
    # Target T[a, b, c]
    
    # We can precompute the indices of non-zero entries
    # But the zero entries are also important.
    # So we use all entries.
    
    # To speed up, we can precompute the list of (a,b) pairs for each c.
    # But since we are using full least squares, we use all (a,b) pairs.
    
    # Precompute mapping from flat indices to (i,j,k)
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    # Precompute the target tensor as a dense array?
    # Size: 9 * 12 * 12 = 1296. Small enough.
    T = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[u_idx(i,j), v_idx(j,k), w_idx(k,i)] = 1.0

    def verify(U, V, W, tol=1e-8):
        # Check all entries
        # Reconstruct T_recon
        # T_recon[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # We can do this efficiently using einsum or loops
        # For small dims, loops are fine.
        for a in range(dim_U):
            for b in range(dim_V):
                for c in range(dim_W):
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[a, r] * V[b, r] * W[c, r]
                    if abs(val - T[a, b, c]) > tol:
                        return False
        return True

    def als_step(U, V, W, R, max_iters=100, tol=1e-12):
        # Perform ALS iterations
        # Precompute structures?
        
        # For efficiency, we can precompute the non-zero indices of T
        # But we are solving for all entries.
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Update W
            # For each c, solve M_c W_c = T_c
            # M_c has rows (a,b), cols r. M_c[(a,b), r] = U[a,r] V[b,r]
            # T_c is a vector of length dim_U * dim_V.
            for c in range(dim_W):
                # Build M_c
                # M_c is (dim_U * dim_V) x R
                # Flatten a,b
                M = np.zeros((dim_U * dim_V, R), dtype=np.float64)
                # T_c is flattened
                T_c = T[:, :, c].flatten()
                
                idx = 0
                for a in range(dim_U):
                    for b in range(dim_V):
                        M[idx, :] = U[a, :] * V[b, :]
                        idx += 1
                
                # Solve least squares
                # M W_c = T_c
                try:
                    W[c, :] = np.linalg.lstsq(M, T_c, rcond=None)[0]
                except:
                    W[c, :] = 0.0
                    
            # Update V
            for b in range(dim_V):
                M = np.zeros((dim_U * dim_W, R), dtype=np.float64)
                T_b = T[:, b, :].flatten()
                idx = 0
                for a in range(dim_U):
                    for c in range(dim_W):
                        M[idx, :] = U[a, :] * W[c, :]
                        idx += 1
                try:
                    V[b, :] = np.linalg.lstsq(M, T_b, rcond=None)[0]
                except:
                    V[b, :] = 0.0

            # Update U
            for a in range(dim_U):
                M = np.zeros((dim_V * dim_W, R), dtype=np.float64)
                T_a = T[a, :, :].flatten()
                idx = 0
                for b in range(dim_V):
                    for c in range(dim_W):
                        M[idx, :] = V[b, :] * W[c, :]
                        idx += 1
                try:
                    U[a, :] = np.linalg.lstsq(M, T_a, rcond=None)[0]
                except:
                    U[a, :] = 0.0

            # Check convergence
            # Calculate error
            # err = sum (T - sum U V W)^2
            # We can check a few random entries or all.
            # All is fast enough (1296 * R ops)
            err = 0.0
            for a in range(dim_U):
                for b in range(dim_V):
                    for c in range(dim_W):
                        val = 0.0
                        for r in range(R):
                            val += U[a, r] * V[b, r] * W[c, r]
                        err += (val - T[a, b, c])**2
            
            if err < tol:
                return True, U, V, W
                
        return False, U, V, W

    best_U = None
    best_V = None
    best_W = None
    best_rank = 36 # Schoolbook

    # Try ranks from low to high
    for R in range(10, 37):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Multiple random restarts
        for restart in range(20):
            if time.time() > deadline - 5:
                break
                
            rng = np.random.RandomState(restart * 1000 + R)
            U = rng.randn(dim_U, R)
            V = rng.randn(dim_V, R)
            W = rng.randn(dim_W, R)
            
            # Normalize to avoid scaling issues
            U /= (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
            V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
            W /= (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
            
            converged, U, V, W = als_step(U, V, W, R, max_iters=200)
            
            if converged:
                # Final verification with tight tolerance
                if verify(U, V, W, tol=1e-9):
                    best_U, best_V, best_W = U, V, W
                    best_rank = R
                    found = True
                    break # Found valid solution for this R, try lower R? No, we go low to high, so this is the best.
                    
        if found:
            break # Found the minimum rank

    # If no better rank found, use schoolbook
    if best_rank == 36 and best_U is None:
        U = np.zeros((dim_U, 36), dtype=np.float64)
        V = np.zeros((dim_V, 36), dtype=np.float64)
        W = np.zeros((dim_W, 36), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[u_idx(i,j), col] = 1.0
                    V[v_idx(j,k), col] = 1.0
                    W[w_idx(k,i), col] = 1.0
                    col += 1
        best_U, best_V, best_W = U, V, W

    def to_list(mat):
        return [[float(mat[i][j]) for j in range(mat.shape[1])] for i in range(mat.shape[0])]

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
