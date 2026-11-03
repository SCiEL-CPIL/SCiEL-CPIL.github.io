import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 120.0
    n, m, p = 2, 3, 6
    
    N_U = n * m
    N_V = m * p
    N_W = p * n
    
    # Construct the target tensor T as a sparse-like structure or dense
    # T[u, v, w] = 1 if indices match, else 0
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    
    # Precompute indices
    # For each u, which (v, w) pairs are non-zero?
    # u -> (i, j)
    # v -> (j', k)
    # w -> (k', i')
    # Non-zero if j == j' and k == k' and i == i'
    
    # Let's just build the dense T if it's small enough
    # Size 6 x 18 x 12 = 1296. Very small.
    T = np.zeros((N_U, N_V, N_W), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    def solve_als(r, max_iter=100, tol=1e-10):
        # Initialize
        np.random.seed(42)
        U = np.random.randn(N_U, r)
        V = np.random.randn(N_V, r)
        W = np.random.randn(N_W, r)
        
        # Normalize to avoid scaling issues
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        prev_err = float('inf')
        
        for it in range(max_iter):
            if time.time() - start_time > DEADLINE - 5:
                break
                
            # Solve for U
            for u in range(N_U):
                # u = i*m + j
                i = u // m
                j = u % m
                
                # Build X: (N_V * N_W) x r
                # X[idx, c] = V[v, c] * W[w, c]
                # y[idx] = T[u, v, w]
                # We only need non-zero entries of T?
                # T is sparse.
                # For a given u, T[u, v, w] is non-zero only for specific v, w.
                # Let's find them.
                # u = i*m + j
                # v must be j*p + k for some k.
                # w must be k*n + i for that k.
                # So for fixed u (i,j), the non-zero pairs (v,w) are:
                # k in 0..p-1
                # v = j*p + k
                # w = k*n + i
                
                # Number of non-zeros for this u is p.
                # We can solve the LSQ using only these rows.
                
                # Construct X_sub (p x r) and y_sub (p x 1)
                X_sub = np.zeros((p, r))
                y_sub = np.zeros(p)
                
                for idx_in, k in enumerate(range(p)):
                    v = j * p + k
                    w = k * n + i
                    # X_sub[idx_in, :] = V[v, :] * W[w, :]
                    X_sub[idx_in, :] = V[v, :] * W[w, :]
                    y_sub[idx_in] = 1.0 # T is 1 here
                
                # Solve X_sub^T X_sub U_u = X_sub^T y_sub
                # Add small regularization for stability
                reg = 1e-8
                A = X_sub.T @ X_sub + reg * np.eye(r)
                b = X_sub.T @ y_sub
                try:
                    U[u, :] = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    U[u, :] = np.linalg.lstsq(A, b, rcond=None)[0]
            
            # Solve for V
            for v in range(N_V):
                # v = j*p + k
                j = v // p
                k = v % p
                # Non-zero u, w for this v:
                # u = i*m + j
                # w = k*n + i
                # i in 0..n-1
                
                X_sub = np.zeros((n, r))
                y_sub = np.zeros(n)
                
                for idx_in, i in enumerate(range(n)):
                    u = i * m + j
                    w = k * n + i
                    X_sub[idx_in, :] = U[u, :] * W[w, :]
                    y_sub[idx_in] = 1.0
                
                A = X_sub.T @ X_sub + reg * np.eye(r)
                b = X_sub.T @ y_sub
                try:
                    V[v, :] = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    V[v, :] = np.linalg.lstsq(A, b, rcond=None)[0]
            
            # Solve for W
            for w in range(N_W):
                # w = k*n + i
                k = w // n
                i = w % n
                # Non-zero u, v for this w:
                # u = i*m + j
                # v = j*p + k
                # j in 0..m-1
                
                X_sub = np.zeros((m, r))
                y_sub = np.zeros(m)
                
                for idx_in, j in enumerate(range(m)):
                    u = i * m + j
                    v = j * p + k
                    X_sub[idx_in, :] = U[u, :] * V[v, :]
                    y_sub[idx_in] = 1.0
                
                A = X_sub.T @ X_sub + reg * np.eye(r)
                b = X_sub.T @ y_sub
                try:
                    W[w, :] = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    W[w, :] = np.linalg.lstsq(A, b, rcond=None)[0]
            
            # Calculate Error
            # Sum (sum_c U V W - T)^2
            # We can compute the reconstructed tensor R
            # R[u, v, w] = sum_c U[u,c] V[v,c] W[w,c]
            # This is a batched matrix multiplication or einsum
            # R = np.einsum('uc,vc,wc->uvw', U, V, W)
            # This might be slow if done every iteration.
            # But N_U=6, N_V=18, N_W=12, r~20.
            # 6*18*12*20 = 25920 ops. Very fast.
            
            R = np.einsum('uc,vc,wc->uvw', U, V, W)
            err = np.sum((R - T)**2)
            
            if err < tol:
                break
            if abs(prev_err - err) < 1e-12:
                break
            prev_err = err
        
        return U, V, W, err

    def verify_and_round(U, V, W, r):
        # Round to nearest 0.5
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Verify exactness
        R = np.einsum('uc,vc,wc->uvw', U_r, V_r, W_r)
        if np.allclose(R, T, atol=1e-8):
            return U_r, V_r, W_r, True
        return None, None, None, False

    best_rank = n * m * p
    best_decomp = None
    
    # Schoolbook
    U_sb = np.zeros((N_U, best_rank))
    V_sb = np.zeros((N_V, best_rank))
    W_sb = np.zeros((N_W, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U_sb[u, col] = 1
                V_sb[v, col] = 1
                W_sb[w, col] = 1
                col += 1
    
    best_decomp = (U_sb, V_sb, W_sb)
    
    # Try to find lower rank
    # Start from low rank and go up?
    # Or start from a guess like 24 and go down?
    # Let's try a range.
    # If we find a valid one at rank r, we can try r-1, etc.
    # But ALS is stochastic.
    # Let's try ranks 20 to 35.
    
    for r in range(20, best_rank):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        U, V, W, err = solve_als(r, max_iter=50)
        
        if err < 1e-8:
            U_r, V_r, W_r, valid = verify_and_round(U, V, W, r)
            if valid:
                best_rank = r
                best_decomp = (U_r, V_r, W_r)
                # If we found a very low rank, we might want to continue searching for lower
                # But let's just keep the best found so far.
                # We can break if we reach a theoretical lower bound?
                # Lower bound is max(n*m, m*p, p*n, n*p) = 18?
                # Actually rank >= max(N_U, N_V, N_W) = 18.
                if r <= 18:
                    break
        else:
            # Even if err is not small, maybe rounding works? Unlikely.
            pass

    return {"U": best_decomp[0].tolist(), "V": best_decomp[1].tolist(), "W": best_decomp[2].tolist()}
