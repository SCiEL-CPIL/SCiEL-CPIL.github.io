"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 54

    # Build the target tensor T[a, b, c] where:
    # a = i*m + j (row i, col j of A), i in [0,n), j in [0,m)
    # b = j*p + k (row j, col k of B), j in [0,m), k in [0,p)
    # c = k*n + i (output index), k in [0,p), i in [0,n)
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook decomposition as fallback
    def schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = schoolbook()

    def to_list(mat):
        # Convert to list of lists, rounding to nearest half-integer
        result = []
        for row in mat:
            r = []
            for val in row:
                rv = round(val * 2) / 2
                r.append(rv)
            result.append(r)
        return result

    def verify_fast(U, V, W, tol=1e-9):
        """Faster verification using matrix operations."""
        R = U.shape[1]
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(T_recon - T)) < tol

    def compute_loss(U_mat, V_mat, W_mat):
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = U_mat[i * m + j, :]
                    v = V_mat[j * p + k, :]
                    w = W_mat[k * n + i, :]
                    val = np.dot(u * v, w)
                    loss += (val - 1.0) ** 2
        return loss

    def als_step(U_mat, V_mat, W_mat, rank, iters=100, tol=1e-12):
        """
        Alternating least squares: fix two matrices, solve for the third.
        Returns (U_mat, V_mat, W_mat, loss).
        """
        for it in range(iters):
            if time.time() > deadline - 5:
                break

            # Solve for U
            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    M = np.zeros((rank, rank))
                    for k in range(p):
                        M += np.outer(V_mat[j * p + k, :], W_mat[k * n + i, :])
                    try:
                        U_new[i * m + j, :] = np.linalg.solve(M, np.ones(rank))
                    except np.linalg.LinAlgError:
                        U_new[i * m + j, :] = np.linalg.lstsq(M, np.ones(rank), rcond=None)[0]
            U_mat = U_new

            # Solve for V
            V_new = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    M = np.zeros((rank, rank))
                    for i in range(n):
                        M += np.outer(U_mat[i * m + j, :], W_mat[k * n + i, :])
                    try:
                        V_new[j * p + k, :] = np.linalg.solve(M, np.ones(rank))
                    except np.linalg.LinAlgError:
                        V_new[j * p + k, :] = np.linalg.lstsq(M, np.ones(rank), rcond=None)[0]
            V_mat = V_new

            # Solve for W
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    M = np.zeros((rank, rank))
                    for j in range(m):
                        M += np.outer(U_mat[i * m + j, :], V_mat[j * p + k, :])
                    try:
                        W_new[k * n + i, :] = np.linalg.solve(M, np.ones(rank))
                    except np.linalg.LinAlgError:
                        W_new[k * n + i, :] = np.linalg.lstsq(M, np.ones(rank), rcond=None)[0]
            W_mat = W_new

            loss = compute_loss(U_mat, V_mat, W_mat)
            if loss < tol:
                break

        loss = compute_loss(U_mat, V_mat, W_mat)
        return U_mat, V_mat, W_mat, loss

    def polish(U_mat, V_mat, W_mat, rank, iters=200):
        """
        High-precision polish: run ALS with tighter tolerance.
        """
        for it in range(iters):
            if time.time() > deadline - 3:
                break

            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    M = np.zeros((rank, rank))
                    for k in range(p):
                        M += np.outer(V_mat[j * p + k, :], W_mat[k * n + i, :])
                    try:
                        U_new[i * m + j, :] = np.linalg.solve(M, np.ones(rank))
                    except np.linalg.LinAlgError:
                        U_new[i * m + j, :] = np.linalg.lstsq(M, np.ones(rank), rcond=None)[0]
            U_mat = U_new

            V_new = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    M = np.zeros((rank, rank))
                    for i in range(n):
                        M += np.outer(U_mat[i * m + j, :], W_mat[k * n + i, :])
                    try:
                        V_new[j * p + k, :] = np.linalg.solve(M, np.ones(rank))
                    except np.linalg.LinAlgError:
                        V_new[j * p + k, :] = np.linalg.lstsq(M, np.ones(rank), rcond=None)[0]
            V_mat = V_new

            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    M = np.zeros((rank, rank))
                    for j in range(m):
                        M += np.outer(U_mat[i * m + j, :], V_mat[j * p + k, :])
                    try:
                        W_new[k * n + i, :] = np.linalg.solve(M, np.ones(rank))
                    except np.linalg.LinAlgError:
                        W_new[k * n + i, :] = np.linalg.lstsq(M, np.ones(rank), rcond=None)[0]
            W_mat = W_new

            loss = compute_loss(U_mat, V_mat, W_mat)
            if loss < 1e-14:
                break

        return U_mat, V_mat, W_mat

    def try_rank(rank, max_attempts=20):
        """
        Try to find a rank-`rank` decomposition.
        Returns (U, V, W) if found, else None.
        """
        if time.time() > deadline - 10:
            return None

        best_result = None
        best_loss = float('inf')

        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break

            np.random.seed(rank * 1000 + attempt * 7 + 42)

            # Try different initializations
            if attempt % 4 == 0:
                # Random initialization
                U = np.random.randn(n * m, rank) * 0.5
                V = np.random.randn(m * p, rank) * 0.5
                W = np.random.randn(p * n, rank) * 0.5
            elif attempt % 4 == 1:
                # Structured: group by i
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                W = np.zeros((p * n, rank))
                terms_per_i = rank // n
                for i in range(n):
                    for t in range(terms_per_i):
                        r = i * terms_per_i + t
                        j = t % m
                        k = (t // m) % p
                        U[i * m + j, r] = 1.0
                        V[j * p + k, r] = 1.0
                        W[k * n + i, r] = 1.0
                U += np.random.randn(n * m, rank) * 0.3
                V += np.random.randn(m * p, rank) * 0.3
                W += np.random.randn(p * n, rank) * 0.3
            elif attempt % 4 == 2:
                # Schoolbook subset with perturbation
                U = U_sb[:, :rank].copy()
                V = V_sb[:, :rank].copy()
                W = W_sb[:, :rank].copy()
                U += np.random.randn(n * m, rank) * 0.2
                V += np.random.randn(m * p, rank) * 0.2
                W += np.random.randn(p * n, rank) * 0.2
            else:
                # Block-structured
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                W = np.zeros((p * n, rank))
                R_per_i = rank // n
                for i in range(n):
                    for r in range(R_per_i):
                        col = i * R_per_i + r
                        j = r % m
                        k = (r // m) % p
                        U[i * m + j, col] = 1.0
                        V[j * p + k, col] = 1.0
                        W[k * n + i, col] = 1.0
                U += np.random.randn(n * m, rank) * 0.2
                V += np.random.randn(m * p, rank) * 0.2
                W += np.random.randn(p * n, rank) * 0.2

            # Run ALS
            U, V, W, loss = als_step(U, V, W, rank, iters=80)

            if loss < best_loss:
                best_loss = loss
                best_result = (U.copy(), V.copy(), W.copy())

            # If loss is very small, try polishing
            if loss < 1e-6:
                U, V, W = polish(U, V, W, rank, iters=100)
                loss = compute_loss(U, V, W)
                if loss < 1e-10:
                    # Check if close to half-integers
                    if np.max(np.abs(U * 2 - np.round(U * 2))) < 0.15 and \
                       np.max(np.abs(V * 2 - np.round(V * 2))) < 0.15 and \
                       np.max(np.abs(W * 2 - np.round(W * 2))) < 0.15:
                        U_rounded = np.round(U * 2) / 2
                        V_rounded = np.round(V * 2) / 2
                        W_rounded = np.round(W * 2) / 2
                        if verify_fast(U_rounded, V_rounded, W_rounded, tol=1e-6):
                            return (U_rounded, V_rounded, W_rounded)

        # Try rounding the best result
        if best_result is not None and best_loss < 1e-4:
            U, V, W = best_result
            for scale in [2, 4]:
                U_rounded = np.round(U * scale) / scale
                V_rounded = np.round(V * scale) / scale
                W_rounded = np.round(W * scale) / scale
                if verify_fast(U_rounded, V_rounded, W_rounded, tol=1e-6):
                    return (U_rounded, V_rounded, W_rounded)

        return None

    best_rank = n * m * p
    best_result = (to_list(U_sb), to_list(V_sb), to_list(W_sb))

    # Try to find lower rank decompositions
    # Start from rank 15 (3 blocks of 2x3x3 with rank 5 each) and go up
    for R in range(15, min(54, n * m * p)):
        if time.time() > deadline - 20:
            break

        result = try_rank(R, max_attempts=15)
        if result is not None:
            best_rank = R
            U_found, V_found, W_found = result
            best_result = (to_list(U_found), to_list(V_found), to_list(W_found))
            break

    # Final verification
    U_final, V_final, W_final = best_result

    U_np = np.array(U_final, dtype=float)
    V_np = np.array(V_final, dtype=float)
    W_np = np.array(W_final, dtype=float)

    if not verify_fast(U_np, V_np, W_np, tol=1e-9):
        # Fall back to schoolbook
        U_np, V_np, W_np = U_sb, V_sb, W_sb
        U_final, V_final, W_final = to_list(U_np), to_list(V_np), to_list(W_np)

    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
