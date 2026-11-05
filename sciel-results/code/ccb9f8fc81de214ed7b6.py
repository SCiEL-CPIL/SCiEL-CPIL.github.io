import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()

    def verify(U, V, W):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        expected = 1.0 if j1 == j2 else 0.0
                        actual = 0.0
                        for c in range(U.shape[1]):
                            actual += U[i * m + j1, c] * V[j2 * p + k, c] * W[k * n + i, c]
                        if abs(actual - expected) > 1e-6:
                            return False
        return True

    def check_and_solve(U, V, r):
        """
        U: shape (n, m, r)
        V: shape (m, p, r)
        Returns W (p, n, r) if valid, else None.
        """
        W = np.zeros((p, n, r))
        b_ik = np.eye(m).flatten()
        for i in range(n):
            for k in range(p):
                U_i = U[i, :, :]  # m x r
                V_k_mat = V[:, k, :].T  # r x m

                # Construct A_ik (m^2 x r)
                A_ik = np.zeros((m * m, r))
                for c in range(r):
                    A_ik[:, c] = np.kron(V_k_mat[c, :].T, U_i[:, c])

                try:
                    w_ik, residuals, rank_val, s = np.linalg.lstsq(A_ik, b_ik, rcond=None)
                    if len(residuals) > 0:
                        if residuals[0] > 1e-8:
                            return None
                    else:
                        residual = np.linalg.norm(A_ik @ w_ik - b_ik)
                        if residual > 1e-8:
                            return None
                    W[k, i, :] = w_ik
                except Exception:
                    return None
        return W

    # Search for lower rank
    for r in range(rank_school - 1, 15, -1):
        if time.time() > deadline - 10:
            break

        # More tries for smaller ranks where success is less likely
        if r > 55:
            num_tries = 30
        elif r > 45:
            num_tries = 50
        elif r > 35:
            num_tries = 80
        else:
            num_tries = 120

        found = False
        for _ in range(num_tries):
            if time.time() > deadline - 5:
                break

            U_rand = np.random.randn(n, m, r)
            V_rand = np.random.randn(m, p, r)

            W_rand = check_and_solve(U_rand, V_rand, r)

            if W_rand is not None:
                U_out = U_rand.reshape(n * m, r)
                V_out = V_rand.reshape(m * p, r)
                W_out = W_rand.T.reshape(p * n, r)

                # Final verification
                if verify(U_out, V_out, W_out):
                    best_rank = r
                    best_U = U_out.copy()
                    best_V = V_out.copy()
                    best_W = W_out.copy()
                    found = True
                    break

        if found:
            # Continue searching for even lower rank
            continue
        else:
            # If we couldn't find a decomposition at this rank, try a few more at the next lower rank
            # but stop if time is running out
            if time.time() > deadline - 15:
                break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
