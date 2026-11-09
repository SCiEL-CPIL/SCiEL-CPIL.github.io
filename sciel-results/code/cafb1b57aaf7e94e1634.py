import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    start_time = time.time()
    deadline = start_time + 140  # safety margin

    # Build the target tensor T
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    best_result = None

    for rank in range(30, 91):
        if time.time() > deadline - 5:
            break

        for trial in range(30):
            if time.time() > deadline - 5:
                break

            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)

            # Precompute tensor reshapes
            T_flat = T.reshape(nm * mp, pn)  # T_flat[a*mp+b, c]
            T_u = T.reshape(nm, mp * pn)     # T_u[a, b*pn+c]
            T_v = T.reshape(mp, nm * pn)     # T_v[b, a*pn+c]

            converged = False
            for it in range(150):
                if time.time() > deadline - 5:
                    break

                # Update W
                D = np.zeros((nm * mp, rank))
                for r in range(rank):
                    D[:, r] = (U[:, r][:, None] * V[:, r][None, :]).flatten()
                try:
                    W_new = np.linalg.lstsq(D, T_flat.T, rcond=None)[0].T
                except:
                    break

                # Update U
                D_u = np.zeros((mp * pn, rank))
                for r in range(rank):
                    D_u[:, r] = (V[:, r][:, None] * W_new[:, r][None, :]).flatten()
                try:
                    U_new = np.linalg.lstsq(D_u, T_u.T, rcond=None)[0].T
                except:
                    break

                # Update V
                D_v = np.zeros((nm * pn, rank))
                for r in range(rank):
                    D_v[:, r] = (U_new[:, r][:, None] * W_new[:, r][None, :]).flatten()
                try:
                    V_new = np.linalg.lstsq(D_v, T_v.T, rcond=None)[0].T
                except:
                    break

                # Check convergence
                err = (np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V)) + np.max(np.abs(W_new - W)))
                U, V, W = U_new, V_new, W_new
                if err < 1e-12:
                    converged = True
                    break

            # Check if we have an exact solution
            T_approx = np.zeros((nm, mp, pn), dtype=np.float64)
            for r in range(rank):
                T_approx += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])

            max_err = np.max(np.abs(T_approx - T))
            if max_err < 1e-8:
                # Try rounding to integers first, then half-integers
                for scale in [1, 2]:
                    U_round = np.round(U * scale) / scale
                    V_round = np.round(V * scale) / scale
                    W_round = np.round(W * scale) / scale

                    T_approx_round = np.zeros((nm, mp, pn), dtype=np.float64)
                    for r in range(rank):
                        T_approx_round += np.einsum('a,b,c->abc', U_round[:, r], V_round[:, r], W_round[:, r])

                    max_err_round = np.max(np.abs(T_approx_round - T))
                    if max_err_round < 1e-8:
                        return {
                            "U": U_round.tolist(),
                            "V": V_round.tolist(),
                            "W": W_round.tolist()
                        }

    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
