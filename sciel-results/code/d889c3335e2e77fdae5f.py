import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    deadline = time.time() + 140  # 140s safety margin

    # Build the rank-50 decomposition
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    u_a, u_b, u_c, u_d = 0, 1, 3, 4

    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1

        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1

        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1

        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1

        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1

        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1

        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1

        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        W[w_00][col] += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1

        W[w_01][col+2] += 1
        W[w_01][col+4] += 1

        W[w_10][col+1] += 1
        W[w_10][col+3] += 1

        W[w_11][col] += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1

        col += 7

    k8 = 8
    v_e = k8
    v_g = p + k8

    U[u_a][col] += 1
    V[v_e][col] += 1
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1

    w_08 = k8 * n + 0
    w_18 = k8 * n + 1

    W[w_08][col] += 1
    W[w_08][col+1] += 1
    W[w_18][col+2] += 1
    W[w_18][col+3] += 1

    col += 4

    u_ar0, u_ar1 = 2, 5
    for k in range(9):
        v_b = 2 * p + k
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        col += 2

    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    if time.time() > deadline - 10:
        return {"U": U, "V": V, "W": W}

    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n*m):
        ai = i // m
        am = i % m
        for j in range(m*p):
            bj = j // p
            bm = j % p
            if am == bj:
                k_idx = bm * n + ai
                T[i, j, k_idx] = 1.0

    # Verify current decomposition
    T_decomp = np.einsum('ic,jc,kc->ijk', U_np, V_np, W_np)
    residual = np.linalg.norm(T_decomp - T)

    if residual > 1e-10:
        return {"U": U, "V": V, "W": W}

    # Check for linear dependencies using vectorized tensor flattening
    nm, mp, pn = n*m, m*p, p*n
    total_size = nm * mp * pn

    # Build the matrix where each column is the flattened rank-1 tensor
    mat = np.zeros((total_size, rank))
    for c in range(rank):
        mat[:, c] = np.kron(np.kron(U_np[:, c], V_np[:, c]), W_np[:, c])

    if time.time() > deadline - 20:
        return {"U": U, "V": V, "W": W}

    # SVD to find rank and null space
    try:
        _, s, Vt = np.linalg.svd(mat, full_matrices=False)
        tol = s[0] * max(nm, mp, pn) * 1e-10
        num_rank = np.sum(s > tol)

        if num_rank < rank:
            null_dim = rank - num_rank
            null_vecs = Vt[num_rank:]  # shape (null_dim, rank)

            best_rank = rank
            best_U = U_np.copy()
            best_V = V_np.copy()
            best_W = W_np.copy()

            # Greedy elimination: try removing columns one at a time
            # Use null space information to prioritize which columns to try
            max_passes = 8
            for _pass in range(max_passes):
                if time.time() > deadline - 15:
                    break

                found_any = False
                # Try all columns, but prioritize those with high null-space weight
                col_order = list(range(best_U.shape[1]))
                if null_dim > 0:
                    # Compute null-space weight for each column
                    null_weights = np.sum(null_vecs ** 2, axis=0)
                    col_order.sort(key=lambda c: -null_weights[c])

                for drop_col in col_order:
                    if time.time() > deadline - 15:
                        break

                    U_new = np.delete(best_U, drop_col, axis=1)
                    V_new = np.delete(best_V, drop_col, axis=1)
                    W_new = np.delete(best_W, drop_col, axis=1)

                    T_new = np.einsum('ic,jc,kc->ijk', U_new, V_new, W_new)
                    res = np.linalg.norm(T_new - T)

                    if res < 1e-10:
                        best_U = U_new
                        best_V = V_new
                        best_W = W_new
                        found_any = True
                        # Recompute null space after removal
                        if best_U.shape[1] > 0:
                            mat_new = np.zeros((total_size, best_U.shape[1]))
                            for c in range(best_U.shape[1]):
                                mat_new[:, c] = np.kron(np.kron(best_U[:, c], best_V[:, c]), best_W[:, c])
                            _, s_new, Vt_new = np.linalg.svd(mat_new, full_matrices=False)
                            tol_new = s_new[0] * max(nm, mp, pn) * 1e-10
                            num_rank_new = np.sum(s_new > tol_new)
                            if num_rank_new < best_U.shape[1]:
                                null_dim = best_U.shape[1] - num_rank_new
                                null_vecs = Vt_new[num_rank_new:]
                            else:
                                null_dim = 0
                                null_vecs = None
                        break

                if not found_any:
                    break

            if best_U.shape[1] < rank:
                return {
                    "U": best_U.tolist(),
                    "V": best_V.tolist(),
                    "W": best_W.tolist()
                }
    except Exception:
        pass

    # Fallback: try a more aggressive search with random perturbations
    if time.time() < deadline - 30:
        try:
            # Try to find a valid lower-rank decomposition by solving a linear system
            # For each candidate rank r < 50, try to find U, V, W of rank r
            for target_rank in range(40, 50):
                if time.time() > deadline - 20:
                    break

                # Random sparse initialization
                np.random.seed(42)
                U_try = np.zeros((nm, target_rank))
                V_try = np.zeros((mp, target_rank))
                W_try = np.zeros((pn, target_rank))

                for c in range(target_rank):
                    # Random sparse entries
                    for r in range(nm):
                        if np.random.random() < 0.3:
                            U_try[r, c] = np.random.choice([-1, 1])
                    for r in range(mp):
                        if np.random.random() < 0.3:
                            V_try[r, c] = np.random.choice([-1, 1])
                    for r in range(pn):
                        if np.random.random() < 0.3:
                            W_try[r, c] = np.random.choice([-1, 1])

                # Check if this gives a valid decomposition
                T_try = np.einsum('ic,jc,kc->ijk', U_try, V_try, W_try)
                if np.linalg.norm(T_try - T) < 1e-10:
                    return {
                        "U": U_try.tolist(),
                        "V": V_try.tolist(),
                        "W": W_try.tolist()
                    }
        except Exception:
            pass

    # Fall back to rank-50
    return {"U": U, "V": V, "W": W}
