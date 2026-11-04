import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 2, 3, 4

    # Trivial solution as fallback
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial))
    V_trivial = np.zeros((m * p, rank_trivial))
    W_trivial = np.zeros((p * n, rank_trivial))

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1.0
                V_trivial[j * p + k, col] = 1.0
                W_trivial[k * n + i, col] = 1.0
                col += 1

    def compute_error(U, V, W, n, m, p):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    max_err = max(max_err, abs(val - 1.0))
        return max_err

    def verify_solution(U, V, W, n, m, p):
        if not np.allclose(2.0 * U, np.round(2.0 * U), atol=1e-9):
            return False
        if not np.allclose(2.0 * V, np.round(2.0 * V), atol=1e-9):
            return False
        if not np.allclose(2.0 * W, np.round(2.0 * W), atol=1e-9):
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def als_decompose(R, n, m, p, U_init=None, V_init=None, W_init=None, max_iter=300, tol=1e-12, seed=42):
        rng = np.random.RandomState(seed)
        if U_init is None:
            U = rng.randn(n * m, R) * 0.5
            V = rng.randn(m * p, R) * 0.5
            W = rng.randn(p * n, R) * 0.5
        else:
            U = U_init.copy()
            V = V_init.copy()
            W = W_init.copy()

        prev_err = float('inf')
        for it in range(max_iter):
            # Update U
            for i in range(n):
                for j in range(m):
                    A_mat = np.zeros((p, R))
                    for k in range(p):
                        A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(A_mat, b, rcond=None)[0]
                    except:
                        pass

            # Update V
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((n, R))
                    for i in range(n):
                        A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(A_mat, b, rcond=None)[0]
                    except:
                        pass

            # Update W
            for k in range(p):
                for i in range(n):
                    A_mat = np.zeros((m, R))
                    for j in range(m):
                        A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(A_mat, b, rcond=None)[0]
                    except:
                        pass

            err = compute_error(U, V, W, n, m, p)
            if err < tol:
                break
            if abs(prev_err - err) < 1e-15 and it > 50:
                break
            prev_err = err
            if time.time() > deadline - 5:
                break

        return U, V, W, err

    def get_init_from_trivial(R, n, m, p, seed):
        rng = np.random.RandomState(seed)
        # Select R random columns from the trivial solution
        # This is a crude initialization but might help if the solution is a subset
        # However, trivial columns are orthogonal in a sense, so this might not be great.
        # Better: Random linear combinations of trivial columns?
        
        # Let's try a different initialization: 
        # Start with random matrices but scale them to match the norm of trivial solution
        U = rng.randn(n * m, R)
        V = rng.randn(m * p, R)
        W = rng.randn(p * n, R)
        
        # Normalize
        U = U / np.linalg.norm(U, axis=0, keepdims=True)
        V = V / np.linalg.norm(V, axis=0, keepdims=True)
        W = W / np.linalg.norm(W, axis=0, keepdims=True)
        
        return U, V, W

    # Try ranks from 16 upward
    best_solution = None
    for R in range(16, min(rank_trivial + 1, 25)):
        if time.time() > deadline - 10:
            break

        found = False
        # More restarts for lower ranks
        num_seeds = 50 if R < 20 else 20
        
        for seed in range(num_seeds):
            if time.time() > deadline - 5:
                break

            U, V, W = get_init_from_trivial(R, n, m, p, seed)
            U, V, W, err = als_decompose(R, n, m, p, U_init=U, V_init=V, W_init=W, max_iter=200, tol=1e-10, seed=seed)

            if err < 1e-8:
                # Round to half-integers
                U_r = np.round(2.0 * U) / 2.0
                V_r = np.round(2.0 * V) / 2.0
                W_r = np.round(2.0 * W) / 2.0

                if verify_solution(U_r, V_r, W_r, n, m, p):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                found = True

        if found:
            continue
        if time.time() > deadline - 5:
            break

    # Return trivial solution as fallback
    return {"U": U_trivial.tolist(), "V": V_trivial.tolist(), "W": W_trivial.tolist()}
