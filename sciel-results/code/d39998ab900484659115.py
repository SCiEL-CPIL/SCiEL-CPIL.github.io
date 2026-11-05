import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def _check_exact(U, V, W, n, m, p):
    """Check exact tensor decomposition: sum_r U[i*m+j,r]*V[j2*p+k,r]*W[k2*n+i2,r] = delta(i,i2)*delta(j,j2)*delta(k,k2)"""
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    for k2 in range(p):
                        for i2 in range(n):
                            val = np.einsum('r,r,r->', U[i*m+j], V[j2*p+k], W[k2*n+i2])
                            expected = 1.0 if (i==i2 and j==j2 and k==k2) else 0.0
                            if abs(val - expected) > 1e-9:
                                return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Naive solution
    naive_rank = n * m * p
    U_naive = np.zeros((n*m, naive_rank))
    V_naive = np.zeros((m*p, naive_rank))
    W_naive = np.zeros((p*n, naive_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i*m+j, col] = 1
                V_naive[j*p+k, col] = 1
                W_naive[k*n+i, col] = 1
                col += 1

    best_rank = naive_rank
    best_U = U_naive.copy()
    best_V = V_naive.copy()
    best_W = W_naive.copy()

    for target_rank in range(12, naive_rank):
        if time.time() > deadline - 25:
            break

        for restart in range(3):
            if time.time() > deadline - 25:
                break

            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)

            for it in range(200):
                if time.time() > deadline - 25:
                    break

                # Update U: for each (i,j), solve linear system
                U_new = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        # Equations: for all (j2,k,k2,i2): sum_r U[i*m+j,r]*V[j2*p+k,r]*W[k2*n+i2,r] = 1 if i2==i,j2==j,k2==k else 0
                        n_eq = m * p * p * n
                        A_mat = np.zeros((n_eq, target_rank))
                        b_vec = np.zeros(n_eq)
                        idx = 0
                        for j2 in range(m):
                            for k in range(p):
                                for k2 in range(p):
                                    for i2 in range(n):
                                        A_mat[idx, :] = V[j2*p+k] * W[k2*n+i2]
                                        if i2 == i and j2 == j and k2 == k:
                                            b_vec[idx] = 1.0
                                        idx += 1
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        U_new[i*m+j] = sol
                U = U_new

                # Update V: for each (j,k), solve linear system
                V_new = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        n_eq = n * m * p * p
                        A_mat = np.zeros((n_eq, target_rank))
                        b_vec = np.zeros(n_eq)
                        idx = 0
                        for i in range(n):
                            for j2 in range(m):
                                for k2 in range(p):
                                    for i2 in range(n):
                                        A_mat[idx, :] = U[i*m+j2] * W[k2*n+i2]
                                        if j2 == j and k2 == k and i == i2:
                                            b_vec[idx] = 1.0
                                        idx += 1
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        V_new[j*p+k] = sol
                V = V_new

                # Update W: for each (k,i), solve linear system
                W_new = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        n_eq = n * m * p * p
                        A_mat = np.zeros((n_eq, target_rank))
                        b_vec = np.zeros(n_eq)
                        idx = 0
                        for i2 in range(n):
                            for j in range(m):
                                for k2 in range(p):
                                    for j2 in range(m):
                                        A_mat[idx, :] = U[i2*m+j2] * V[j2*p+k2]
                                        if i2 == i and k2 == k and j == j2:
                                            b_vec[idx] = 1.0
                                        idx += 1
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        W_new[k*n+i] = sol
                W = W_new

            if _check_exact(U, V, W, n, m, p):
                U_snap = np.round(U * 2) / 2.0
                V_snap = np.round(V * 2) / 2.0
                W_snap = np.round(W * 2) / 2.0
                if _check_exact(U_snap, V_snap, W_snap, n, m, p):
                    best_rank = target_rank
                    best_U = U_snap
                    best_V = V_snap
                    best_W = W_snap
                    break
            if best_rank < naive_rank:
                break
        if best_rank < naive_rank:
            break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
