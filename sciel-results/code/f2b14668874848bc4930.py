import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    def check_decomposition(U, V, W):
        # Check if sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        # Vectorized check
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # We need to compute T[i,j,k] = sum_c ...
        # This is a bit tricky to vectorize fully without loops over i,j,k if r is large, but n,m,p are small.
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j, :]
                    v_val = V[j * p + k, :]
                    w_val = W[k * n + i, :]
                    prod = np.dot(u_val, v_val * w_val)
                    error += (prod - 1.0) ** 2
        return np.sqrt(error)

    def snap_to_half_integers(U, V, W):
        U_snapped = np.round(U * 2) / 2
        V_snapped = np.round(V * 2) / 2
        W_snapped = np.round(W * 2) / 2
        return U_snapped, V_snapped, W_snapped

    def als_step(U, V, W, factor, iters=10):
        r = U.shape[1]
        for _ in range(iters):
            if factor == 'U':
                for i in range(n):
                    for j in range(m):
                        # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        A = np.zeros((p, r))
                        b = np.ones(p)
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        # Solve A x = b
                        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        U[i * m + j, :] = x
            elif factor == 'V':
                for j in range(m):
                    for k in range(p):
                        # A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        A = np.zeros((n, r))
                        b = np.ones(n)
                        for i in range(n):
                            A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        V[j * p + k, :] = x
            elif factor == 'W':
                for k in range(p):
                    for i in range(n):
                        # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        A = np.zeros((m, r))
                        b = np.ones(m)
                        for j in range(m):
                            A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        W[k * n + i, :] = x
        return U, V, W

    def try_rank(r, max_restarts=100):
        if r >= n * m * p:
            return None
        for restart in range(max_restarts):
            # Random initialization
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            # Run ALS
            for _ in range(20):
                U, V, W = als_step(U, V, W, 'U', iters=1)
                U, V, W = als_step(U, V, W, 'V', iters=1)
                U, V, W = als_step(U, V, W, 'W', iters=1)
                
            err = check_decomposition(U, V, W)
            if err < 1e-5:
                # Snap and verify
                U_s, V_s, W_s = snap_to_half_integers(U, V, W)
                err_s = check_decomposition(U_s, V_s, W_s)
                if err_s < 1e-10:
                    return U_s, V_s, W_s
            # If not exact, try a few more iterations from current state?
            # Or just restart.
            
        return None

    start_time = time.time()
    time_limit = 140.0 # Safety margin
    
    # Try ranks from 23 down to 19
    for r in range(23, 18, -1):
        if time.time() - start_time > time_limit:
            break
        result = try_rank(r, max_restarts=50)
        if result is not None:
            U, V, W = result
            # Convert to lists of lists as required by interface?
            # The interface says: {'U', 'V', 'W'}: factor matrices...
            # The seed returns lists of lists.
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # Fallback to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
