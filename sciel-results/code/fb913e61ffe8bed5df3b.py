"""Matrix multiplication 2,3,4 rank-minimal decomposition search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 110.0  # safety margin before 120s limit

    # Schoolbook (rank 24) as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank

    # Build the list of constrained equations
    # Each equation: (i, k, j, jp) with target delta_{j,jp}
    # U-index a = i*m+j, V-index b = jp*p+k, W-index c = k*n+i
    equations = []
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for jp in range(m):
                    a = i * m + j
                    b = jp * p + k
                    c = k * n + i
                    target = 1.0 if j == jp else 0.0
                    equations.append((a, b, c, target))
    eq_array = np.array([(a, b, c) for (a, b, c, t) in equations])
    targets = np.array([t for (a, b, c, t) in equations])

    def tensor_values(U, V, W):
        # Compute sum_r U[a,r] V[b,r] W[c,r] for all equations
        # U[a, :] @ (V[b, :] * W[c, :]) ... 
        # For each equation (a,b,c): dot(U[a,:], V[b,:] * W[c,:])
        vals = np.einsum('ar,br,cr->r', U, V, W)  # no, that's not right
        # Let me do it explicitly
        result = np.zeros(len(equations))
        for idx, (a, b, c) in enumerate(eq_array):
            result[idx] = np.dot(U[a, :], V[b, :] * W[c, :])
        return result

    def error(U, V, W):
        vals = tensor_values(U, V, W)
        return np.sum((vals - targets) ** 2)

    def verify_exact(U, V, W):
        # U, V, W are half-integers (floats). Check exactness with integer arithmetic.
        U2 = np.round(2.0 * U).astype(np.int64)
        V2 = np.round(2.0 * V).astype(np.int64)
        W2 = np.round(2.0 * W).astype(np.int64)
        # Check: sum_r U2[a,r] V2[b,r] W2[c,r] == 8 * target
        for idx, (a, b, c) in enumerate(eq_array):
            s = int(np.dot(U2[a, :], V2[b, :] * W2[c, :]))
            if s != 8 * int(targets[idx]):
                return False
        return True

    def als_decompose(r, init, max_iter=200, tol=1e-12):
        U, V, W = init
        prev_err = np.inf
        for it in range(max_iter):
            # Update U
            U_new = U.copy()
            for i in range(n):
                # Unknowns: U[i*m:(i+1)*m, :] (m*r variables)
                A = []
                b = []
                for k in range(p):
                    for j in range(m):
                        for jp in range(m):
                            row = np.zeros(m * r)
                            for rprime in range(r):
                                row[j * r + rprime] = W[k * n + i, rprime] * V[jp * p + k, rprime]
                            A.append(row)
                            b.append(1.0 if j == jp else 0.0)
                A_arr = np.array(A)
                b_arr = np.array(b)
                x, _, _, _ = np.linalg.lstsq(A_arr, b_arr, rcond=None)
                U_new[i * m:(i + 1) * m, :] = x.reshape(m, r)
            U = U_new
            # Update V
            V_new = V.copy()
            for jp in range(m):
                # Unknowns: V[jp*p:(jp+1)*p, :] (p*r variables)
                A = []
                b = []
                for i in range(n):
                    for k in range(p):
                        for j in range(m):
                            row = np.zeros(p * r)
                            for rprime in range(r):
                                row[k * r + rprime] = U[i * m + j, rprime] * W[k * n + i, rprime]
                            A.append(row)
                            b.append(1.0 if j == jp else 0.0)
                A_arr = np.array(A)
                b_arr = np.array(b)
                x, _, _, _ = np.linalg.lstsq(A_arr, b_arr, rcond=None)
                V_new[jp * p:(jp + 1) * p, :] = x.reshape(p, r)
            V = V_new
            # Update W
            W_new = W.copy()
            for k in range(p):
                # Unknowns: W[k*n:(k+1)*n, :] (n*r variables)
                A = []
                b = []
                for i in range(n):
                    for j in range(m):
                        for jp in range(m):
                            row = np.zeros(n * r)
                            for rprime in range(r):
                                row[i * r + rprime] = U[i * m + j, rprime] * V[jp * p + k, rprime]
                            A.append(row)
                            b.append(1.0 if j == jp else 0.0)
                A_arr = np.array(A)
                b_arr = np.array(b)
                x, _, _, _ = np.linalg.lstsq(A_arr, b_arr, rcond=None)
                W_new[k * n:(k + 1) * n, :] = x.reshape(n, r)
            W = W_new
            # Check convergence
            err = error(U, V, W)
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        return U, V, W, error(U, V, W)

    best = None  # (rank, U, V, W)

    # Search r from high to low
    for r in range(23, 15, -1):
        if time.time() - start_time > DEADLINE:
            break
        # Try multiple initializations
        found = False
        n_restarts = 0
        max_restarts = 30
        while not found and n_restarts < max_restarts:
            if time.time() - start_time > DEADLINE:
                break
            n_restarts += 1
            # Random init
            scale = 0.5
            U0 = np.random.randn(n * m, r) * scale
            V0 = np.random.randn(m * p, r) * scale
            W0 = np.random.randn(p * n, r) * scale
            U, V, W, err = als_decompose(r, (U0, V0, W0), max_iter=150)
            if verify_exact(U, V, W):
                best = (r, U, V, W)
                found = True
                break
            # Also try with polish
            # (skip for now)
        # Continue to lower r even if found, to try for better
        # But limit total time

    # If nothing found, use schoolbook
    if best is None:
        U, V, W, rank = schoolbook()
    else:
        rank, U, V, W = best

    # Convert to lists of ints/half-ints
    def to_list(M):
        return [[float(x) for x in row] for row in M]

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
