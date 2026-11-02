"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s limit

    # Schoolbook decomposition of rank n*m*p
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=np.float64)
    V_school = np.zeros((m * p, rank_school), dtype=np.float64)
    W_school = np.zeros((p * n, rank_school), dtype=np.float64)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1

    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    best_rank = rank_school

    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def compute_error(U, V, W):
        """Compute sum of squared errors for all tensor entries."""
        r = U.shape[1]
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    err += (s - 1.0) ** 2
        return err

    def alternating_minimize(U, V, W, max_iter=200, tol=1e-12):
        """Alternating least squares for tensor decomposition."""
        r = U.shape[1]
        for it in range(max_iter):
            if time.time() > deadline:
                break
            # Update U: for each (i,j), minimize sum_k (T[i,j,k] - sum_r U[i,j,r]*V[j,k,r]*W[k,i,r])^2
            # This is a linear system in U[i,j,:]
            U_new = U.copy()
            for i in range(n):
                for j in range(m):
                    # Build A[r1, r2] = sum_k V[j,k,r1]*W[k,i,r2]
                    # b[r1] = sum_k W[k,i,r1] (since T[i,j,k]=1)
                    A = np.zeros((r, r))
                    b = np.zeros(r)
                    for k in range(p):
                        vw = np.outer(V[j*p+k], W[k*n+i])
                        A += vw
                        b += W[k*n+i]
                    # Solve A * u = b
                    try:
                        u = np.linalg.solve(A + 1e-12*np.eye(r), b)
                        U_new[i*m+j] = u
                    except np.linalg.LinAlgError:
                        U_new[i*m+j] = np.linalg.lstsq(A + 1e-12*np.eye(r), b, rcond=None)[0]
            U = U_new

            # Update V
            V_new = V.copy()
            for j in range(m):
                for k in range(p):
                    A = np.zeros((r, r))
                    b = np.zeros(r)
                    for i in range(n):
                        uw = np.outer(U[i*m+j], W[k*n+i])
                        A += uw
                        b += U[i*m+j]
                    try:
                        v = np.linalg.solve(A + 1e-12*np.eye(r), b)
                        V_new[j*p+k] = v
                    except np.linalg.LinAlgError:
                        V_new[j*p+k] = np.linalg.lstsq(A + 1e-12*np.eye(r), b, rcond=None)[0]
            V = V_new

            # Update W
            W_new = W.copy()
            for k in range(p):
                for i in range(n):
                    A = np.zeros((r, r))
                    b = np.zeros(r)
                    for j in range(m):
                        uv = np.outer(U[i*m+j], V[j*p+k])
                        A += uv
                        b += U[i*m+j]
                    try:
                        w = np.linalg.solve(A + 1e-12*np.eye(r), b)
                        W_new[k*n+i] = w
                    except np.linalg.LinAlgError:
                        W_new[k*n+i] = np.linalg.lstsq(A + 1e-12*np.eye(r), b, rcond=None)[0]
            W = W_new

            if it > 0:
                err = compute_error(U, V, W)
                if err < tol:
                    break

        return U, V, W, compute_error(U, V, W)

    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = 35, but realistically higher
    # Try ranks from ~60 down to 40
    for r in range(65, 39, -1):
        if time.time() > deadline:
            break

        max_restarts = 50 if r > 55 else 100
        for restart in range(max_restarts):
            if time.time() > deadline:
                break

            # Random initialization
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)

            # Run alternating minimization
            U, V, W, err = alternating_minimize(U, V, W, max_iter=150)

            if err < 1e-10:
                if verify(U, V, W, tol=1e-9):
                    best_rank = r
                    best_U, best_V, best_W = U, V, W
                    break  # Found valid at this rank, try lower

    # Final verification
    if not verify(best_U, best_V, best_W, tol=1e-9):
        # Fallback to schoolbook
        best_U, best_V, best_W = U_school, V_school, W_school
        best_rank = rank_school

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
