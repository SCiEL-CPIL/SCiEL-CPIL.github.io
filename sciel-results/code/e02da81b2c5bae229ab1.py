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

    mat = np.zeros((total_size, rank))
    for c in range(rank):
        mat[:, c] = np.kron(np.kron(U_np[:, c], V_np[:, c]), W_np[:, c])

    if time.time() > deadline - 30:
        return {"U": U, "V": V, "W": W}

    # SVD to find rank and perform aggressive dependency elimination
    try:
        best_U = U_np.copy()
        best_V = V_np.copy()
        best_W = W_np.copy()

        # Iterative SVD-based dependency elimination
        max_outer = 20
        for outer_iter in range(max_outer):
            if time.time() > deadline - 25:
                break

            cur_rank = best_U.shape[1]
            if cur_rank <= 0:
                break

            # Build matrix of flattened rank-1 tensors
            mat = np.zeros((total_size, cur_rank))
            for c in range(cur_rank):
                mat[:, c] = np.kron(np.kron(best_U[:, c], best_V[:, c]), best_W[:, c])

            # SVD to find null space
            _, s, Vt = np.linalg.svd(mat, full_matrices=False)
            tol = s[0] * max(nm, mp, pn) * 1e-9
            num_rank = np.sum(s > tol)

            if num_rank >= cur_rank:
                # No more dependencies found via SVD; try greedy removal
                break

            null_dim = cur_rank - num_rank
            null_vecs = Vt[num_rank:]  # shape (null_dim, cur_rank)

            # Try to find a column that can be eliminated
            # Strategy: for each null vector, find the column with largest coefficient
            # and try removing it
            found_elimination = False

            # Try all null vectors, pick the one that allows removal of a column
            for nv in range(null_dim):
                if time.time() > deadline - 25:
                    break
                coeff = null_vecs[nv]
                # Sort columns by absolute coefficient
                order = np.argsort(-np.abs(coeff))
                for drop_col in order:
                    if time.time() > deadline - 25:
                        break
                    drop_col = int(drop_col)

                    U_new = np.delete(best_U, drop_col, axis=1)
                    V_new = np.delete(best_V, drop_col, axis=1)
                    W_new = np.delete(best_W, drop_col, axis=1)

                    T_new = np.einsum('ic,jc,kc->ijk', U_new, V_new, W_new)
                    res = np.linalg.norm(T_new - T)

                    if res < 1e-10:
                        best_U = U_new
                        best_V = V_new
                        best_W = W_new
                        found_elimination = True
                        break

                if found_elimination:
                    break

            if not found_elimination:
                # Try brute-force greedy: try removing each column
                for drop_col in range(cur_rank):
                    if time.time() > deadline - 25:
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
                        found_elimination = True
                        break

            if not found_elimination:
                break

        # Final verification
        final_rank = best_U.shape[1]
        if final_rank < rank:
            T_final = np.einsum('ic,jc,kc->ijk', best_U, best_V, best_W)
            res_final = np.linalg.norm(T_final - T)
            if res_final < 1e-10:
                return {
                    "U": best_U.tolist(),
                    "V": best_V.tolist(),
                    "W": best_W.tolist()
                }
    except Exception:
        pass

    # Fall back to rank-50
    return {"U": U, "V": V, "W": W}
