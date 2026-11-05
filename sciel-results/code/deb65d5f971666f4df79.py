"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    import numpy as np
    import time

    start_time = time.time()
    deadline = start_time + 130

    # Build the 3x3x4 multiplication tensor
    tensor = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor[a, b, c] = 1.0

    # Try to find a rank-21 decomposition using ALS
    rank = 21
    np.random.seed(42)

    best_error = float('inf')
    best_U = best_V = best_W = None

    for trial in range(10):
        if time.time() > deadline - 20:
            break

        U_mat = np.random.randn(n * m, rank)
        V_mat = np.random.randn(m * p, rank)
        W_mat = np.random.randn(p * n, rank)

        for iteration in range(300):
            if time.time() > deadline - 15:
                break

            # Update U
            for a in range(n * m):
                M_a = np.zeros((rank, m * p * p * n))
                for r in range(rank):
                    M_a[r, :] = (V_mat[:, r] * W_mat[:, r]).flatten()
                U_mat[a, :] = np.linalg.lstsq(M_a, tensor[a].flatten(), rcond=None)[0]

            # Update V
            for b in range(m * p):
                M_b = np.zeros((rank, n * m * p * n))
                for r in range(rank):
                    M_b[r, :] = (U_mat[:, r] * W_mat[:, r]).flatten()
                V_mat[b, :] = np.linalg.lstsq(M_b, tensor[:, b, :].flatten(), rcond=None)[0]

            # Update W
            for c in range(p * n):
                M_c = np.zeros((rank, n * m * m * p))
                for r in range(rank):
                    M_c[r, :] = (U_mat[:, r] * V_mat[:, r]).flatten()
                W_mat[c, :] = np.linalg.lstsq(M_c, tensor[:, :, c].flatten(), rcond=None)[0]

            # Compute error
            pred = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
            error = np.sum((pred - tensor) ** 2)

            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U_mat.copy(), V_mat.copy(), W_mat.copy()

            if error < 1e-12:
                break

        if best_error < 1e-10:
            break

    # Check if we found a valid rank-21 solution
    if best_U is not None and best_error < 1e-8:
        # Try rounding to half-integers
        U_r = np.round(best_U * 2) / 2
        V_r = np.round(best_V * 2) / 2
        W_r = np.round(best_W * 2) / 2

        pred = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        max_err = np.max(np.abs(pred - tensor))

        if max_err < 1e-9:
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

        # Try rounding to integers
        U_i = np.round(best_U)
        V_i = np.round(best_V)
        W_i = np.round(best_W)
        pred = np.einsum('ar,br,cr->abc', U_i, V_i, W_i)
        max_err = np.max(np.abs(pred - tensor))
        if max_err < 1e-9:
            return {"U": U_i.tolist(), "V": V_i.tolist(), "W": W_i.tolist()}

        # Use the unrounded solution (should be close enough)
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Fallback: schoolbook rank 36
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
