"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_school = n * m * p  # 60

    # Build target tensor T[i, j, k] where i=(i1,j1), j=(j2,k2), k=(k3,i3)
    T = np.zeros((nm, mp, pn))
    for i1 in range(n):
        for j1 in range(m):
            for k2 in range(p):
                T[i1 * m + j1, j1 * p + k2, k2 * n + i1] = 1.0

    # Schoolbook decomposition
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i1 in range(n):
        for j1 in range(m):
            for k2 in range(p):
                U_school[i1 * m + j1, col] = 1.0
                V_school[j1 * p + k2, col] = 1.0
                W_school[k2 * n + i1, col] = 1.0
                col += 1

    best_rank = rank_school
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()

    deadline = time.time() + 130  # 10s safety margin

    # Try to find lower rank decompositions
    for r in range(rank_school - 1, max(nm, mp, pn) - 1, -1):
        if time.time() > deadline:
            break

        found = False
        for seed in range(10):
            if time.time() > deadline:
                break
            np.random.seed(seed)

            # Initialize near schoolbook
            U_c = U_school[:, :r].copy() if r <= rank_school else np.random.randn(nm, r)
            V_c = V_school[:, :r].copy() if r <= rank_school else np.random.randn(mp, r)
            W_c = W_school[:, :r].copy() if r <= rank_school else np.random.randn(pn, r)

            # Add small random perturbation
            U_c += 0.05 * np.random.randn(nm, r)
            V_c += 0.05 * np.random.randn(mp, r)
            W_c += 0.05 * np.random.randn(pn, r)

            # Alternating least squares
            for it in range(8):
                if time.time() > deadline:
                    break

                # Solve for U: T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
                M = np.einsum('jr,kr->jkr', V_c, W_c).reshape(mp * pn, r)
                U_c = T.reshape(nm, mp * pn) @ np.linalg.pinv(M.T)

                # Solve for V
                N = np.einsum('ir,kr->ikr', U_c, W_c).reshape(nm * pn, r)
                V_c = T.reshape(mp, nm * pn) @ np.linalg.pinv(N.T)

                # Solve for W
                P = np.einsum('ir,jr->ijr', U_c, V_c).reshape(nm * mp, r)
                W_c = T.reshape(pn, nm * mp) @ np.linalg.pinv(P.T)

            # Check convergence
            T_check = np.einsum('ir,jr,kr->ijk', U_c, V_c, W_c)
            error = np.max(np.abs(T_check - T))

            if error < 1e-8:
                # Try to round to half-integers
                U_r = np.round(U_c * 2) / 2
                V_r = np.round(V_c * 2) / 2
                W_r = np.round(W_c * 2) / 2

                T_check2 = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
                if np.max(np.abs(T_check2 - T)) < 1e-10:
                    best_rank = r
                    best_U, best_V, best_W = U_r, V_r, W_r
                    found = True
                    break

        if found:
            break

    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
