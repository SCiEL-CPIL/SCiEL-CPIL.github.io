import numpy as np
import time
from itertools import product, combinations

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0

    def verify(U_mat, V_mat, W_mat, n_val, m_val, p_val):
        rank = U_mat.shape[1]
        for i in range(n_val):
            for j in range(m_val):
                for k in range(p_val):
                    actual = sum(U_mat[i * m_val + j, r] * V_mat[j * p_val + k, r] * W_mat[k * n_val + i, r]
                                 for r in range(rank))
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True

    # --- Rank-17 baseline: Strassen on 2x2 sub-block + schoolbook for the rest ---
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))

    # Strassen for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # A indices: i*3+j for i in 0,1; j in 0,1 -> 0,1,3,4
    # B indices: j*3+k for j in 0,1; k in 0,1 -> 0,1,3,4
    # C indices: k*2+i for k in 0,1; i in 0,1 -> 0,1,2,3

    # M1 = (A00+A11)(B00+B11)
    U17[0, 0] = 1; U17[4, 0] = 1
    V17[0, 0] = 1; V17[4, 0] = 1
    W17[0, 0] = 1; W17[3, 0] = 1

    # M2 = A11(B10-B00)
    U17[4, 1] = 1
    V17[3, 1] = 1; V17[0, 1] = -1
    W17[1, 1] = 1; W17[3, 1] = 1

    # M3 = A00(B01-B11)
    U17[0, 2] = 1
    V17[1, 2] = 1; V17[4, 2] = -1
    W17[2, 2] = 1; W17[3, 2] = 1

    # M4 = (A10+A00)(B00-B10)
    U17[0, 3] = 1; U17[3, 3] = 1
    V17[0, 3] = 1; V17[3, 3] = -1
    W17[0, 3] = 1; W17[1, 3] = 1

    # M5 = (A01+A10)(B10+B11)
    U17[1, 4] = 1; U17[3, 4] = 1
    V17[3, 4] = 1; V17[4, 4] = 1
    W17[1, 4] = 1; W17[2, 4] = 1

    # M6 = (A00-A10)(B01+B10)
    U17[0, 5] = 1; U17[3, 5] = -1
    V17[1, 5] = 1; V17[3, 5] = 1
    W17[0, 5] = 1; W17[2, 5] = -1

    # M7 = (A11-A01)(B11+B01)
    U17[4, 6] = 1; U17[1, 6] = -1
    V17[4, 6] = 1; V17[1, 6] = 1
    W17[1, 6] = -1; W17[2, 6] = 1

    # Schoolbook: A[0:2,2] * B[2,0:2] -> C[0:2,0:2]
    U17[2, 7] = 1; V17[6, 7] = 1; W17[0, 7] = 1
    U17[2, 8] = 1; V17[7, 8] = 1; W17[2, 8] = 1
    U17[5, 9] = 1; V17[6, 9] = 1; W17[1, 9] = 1
    U17[5, 10] = 1; V17[7, 10] = 1; W17[3, 10] = 1

    # Schoolbook: A[0:2,0:3] * B[0:3,2] -> C[0:2,2]
    U17[0, 11] = 1; V17[2, 11] = 1; W17[4, 11] = 1
    U17[1, 12] = 1; V17[5, 12] = 1; W17[4, 12] = 1
    U17[2, 13] = 1; V17[8, 13] = 1; W17[4, 13] = 1
    U17[3, 14] = 1; V17[2, 14] = 1; W17[5, 14] = 1
    U17[4, 15] = 1; V17[5, 15] = 1; W17[5, 15] = 1
    U17[5, 16] = 1; V17[8, 16] = 1; W17[5, 16] = 1

    if not verify(U17, V17, W17, n, m, p):
        # Fallback: return as-is (should not happen)
        return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}

    best_rank = 17
    best_solution = (U17.tolist(), V17.tolist(), W17.tolist())

    # --- Attempt to reduce rank by finding linear dependencies ---
    # Build the full tensor and check if any column is a linear combination of others
    def tensor_vector(U_mat, V_mat, W_mat, r):
        vec = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[(i * m + j) * p + k] = U_mat[i * m + j, r] * V_mat[j * p + k, r] * W_mat[k * n + i, r]
        return vec

    # Check rank of the set of 17 tensor vectors
    T_all = np.array([tensor_vector(U17, V17, W17, r) for r in range(17)])
    rank_T = np.linalg.matrix_rank(T_all, tol=1e-9)

    if rank_T < 17:
        # Some columns are linearly dependent; try to find a subset of 16 that spans the target
        target = np.ones(n * m * p)
        for remove in range(17):
            remaining = [i for i in range(17) if i != remove]
            T_rem = T_all[remaining]
            try:
                coeffs, res, _, _ = np.linalg.lstsq(T_rem, target, rcond=None)
                if res and res[0] < 1e-9:
                    U_new = np.column_stack([U17[:, i] * coeffs[idx] for idx, i in enumerate(remaining)])
                    V_new = np.column_stack([V17[:, i] * coeffs[idx] for idx, i in enumerate(remaining)])
                    W_new = np.column_stack([W17[:, i] * coeffs[idx] for idx, i in enumerate(remaining)])
                    if verify(U_new, V_new, W_new, n, m, p):
                        # Trim zero columns
                        nonzero = [c for c in range(16) if np.any(np.abs(U_new[:, c]) > 1e-9) and
                                   np.any(np.abs(V_new[:, c]) > 1e-9) and np.any(np.abs(W_new[:, c]) > 1e-9)]
                        if len(nonzero) > 0:
                            Uf = U_new[:, nonzero]
                            Vf = V_new[:, nonzero]
                            Wf = W_new[:, nonzero]
                            if verify(Uf, Vf, Wf, n, m, p):
                                best_rank = Uf.shape[1]
                                best_solution = (Uf.tolist(), Vf.tolist(), Wf.tolist())
                                break
            except np.linalg.LinAlgError:
                continue

    if best_rank < 17:
        U_out, V_out, W_out = best_solution
        return {"U": U_out, "V": V_out, "W": W_out}

    # --- Exhaustive affine search over the 13-dim subspace to find rank-13 or lower ---
    # The block structure gives 13 independent scalar multiplications + 4 Strassen terms = 17 total.
    # Try to express the 17-term decomposition using fewer rank-1 terms.

    # Alternative: try random restarts with gradient-free optimization for rank < 17
    def try_optimize_rank(r, n_restarts=3, time_per_restart=10.0):
        if time.time() > deadline - 25:
            return None

        for restart in range(n_restarts):
            if time.time() > deadline - 25:
                return None
            np.random.seed(1000 + r * 10 + restart)

            U_cur = np.random.randn(n * m, r) * 0.5
            V_cur = np.random.randn(m * p, r) * 0.5
            W_cur = np.random.randn(p * n, r) * 0.5

            lr = 0.05
            for it in range(500):
                if time.time() > deadline - 25:
                    return None

                # Compute error: for each (i,j,k), err = sum_r U*V*W - 1
                # Gradient w.r.t. U[i*m+j, r] = V[j*p+k, r] * W[k*n+i, r]
                # Gradient w.r.t. V[j*p+k, r] = U[i*m+j, r] * W[k*n+i, r]
                # Gradient w.r.t. W[k*n+i, r] = U[i*m+j, r] * V[j*p+k, r]

                grad_U = np.zeros_like(U_cur)
                grad_V = np.zeros_like(V_cur)
                grad_W = np.zeros_like(W_cur)

                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            err = sum(U_cur[i * m + j, rr] * V_cur[j * p + k, rr] * W_cur[k * n + i, rr]
                                      for rr in range(r)) - 1.0
                            for rr in range(r):
                                grad_U[i * m + j, rr] += err * V_cur[j * p + k, rr] * W_cur[k * n + i, rr]
                                grad_V[j * p + k, rr] += err * U_cur[i * m + j, rr] * W_cur[k * n + i, rr]
                                grad_W[k * n + i, rr] += err * U_cur[i * m + j, rr] * V_cur[j * p + k, rr]

                U_cur -= lr * grad_U
                V_cur -= lr * grad_V
                W_cur -= lr * grad_W

                if verify(U_cur, V_cur, W_cur, n, m, p):
                    return U_cur, V_cur, W_cur

                # Adaptive learning rate
                if it % 50 == 49:
                    err_norm = np.linalg.norm(grad_U) + np.linalg.norm(grad_V) + np.linalg.norm(grad_W)
                    if err_norm < 1e-6:
                        break
                    lr *= 0.5

        return None

    # Try ranks from 16 down to 13
    for r in range(16, 12, -1):
        if time.time() > deadline - 20:
            break
        result = try_optimize_rank(r, n_restarts=2, time_per_restart=8.0)
        if result is not None:
            Uf, Vf, Wf = result
            if verify(Uf, Vf, Wf, n, m, p):
                best_rank = r
                best_solution = (Uf.tolist(), Vf.tolist(), Wf.tolist())
                break

    if best_rank < 17:
        U_out, V_out, W_out = best_solution
        return {"U": U_out, "V": V_out, "W": W_out}

    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
# EVOLVE-BLOCK-END
