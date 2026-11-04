import time
import numpy as np
from fractions import Fraction
from itertools import combinations

def construct():
    n, m, p = 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8
    N = nm * mp * pn  # 1024

    start_time = time.time()
    deadline = start_time + 130.0

    def time_ok():
        return time.time() < deadline - 2.0

    # Build the multiplication tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for l in range(m):
            for k in range(p):
                a = i * m + l
                b = l * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    T_flat = T.flatten()

    def verify_decomp(U_mat, V_mat, W_mat, rank_val):
        S = np.zeros((nm, mp, pn), dtype=float)
        for r in range(rank_val):
            S += np.einsum('a,b,c->abc', U_mat[:, r], V_mat[:, r], W_mat[:, r])
        return np.max(np.abs(S - T))

    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                rounded = round(val * 2) / 2.0
                if abs(rounded - val) < 1e-10:
                    row_list.append(rounded)
                else:
                    row_list.append(val)
            result.append(row_list)
        return result

    # Known rank-28 solution via Strassen on 2x2 blocks
    rank = 28
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        # Standard Strassen 2x2x2 with 7 multiplications
        # M1 = (A00+A11)(B00+B11)
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1
        # M2 = (A10+A11)B00
        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1
        # M3 = A00(B10-B11)
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1
        # M4 = A11(B01-B00)
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1
        # M5 = (A00+A01)B11
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1
        # M6 = (A10-A00)(B00+B01)
        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1
        # M7 = (A01-A11)(B10+B11)
        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1

    # Block 1: A_L B_TL -> C_LL
    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 2, 1, 3]
    add_block(0, X1, Y1, Z1)
    # Block 2: A_R B_BL -> C_LR
    X2 = [2, 3, 6, 7]
    Y2 = [8, 9, 12, 13]
    Z2 = [0, 2, 1, 3]
    add_block(7, X2, Y2, Z2)
    # Block 3: A_L B_TR -> C_RL
    X3 = [0, 1, 4, 5]
    Y3 = [2, 3, 6, 7]
    Z3 = [4, 6, 5, 7]
    add_block(14, X3, Y3, Z3)
    # Block 4: A_R B_BR -> C_RR
    X4 = [2, 3, 6, 7]
    Y4 = [10, 11, 14, 15]
    Z4 = [4, 6, 5, 7]
    add_block(21, X4, Y4, Z4)

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    # Verify rank-28 solution
    max_err = verify_decomp(U_arr, V_arr, W_arr, rank)
    if max_err > 1e-10:
        return {"U": to_list(U_arr), "V": to_list(V_arr), "W": to_list(W_arr)}

    best_rank = rank
    best_U = U_arr.copy()
    best_V = V_arr.copy()
    best_W = W_arr.copy()

    def try_remove_set(U_cur, V_cur, W_cur, cols_to_remove):
        remaining = [c for c in range(U_cur.shape[1]) if c not in cols_to_remove]
        if len(remaining) == 0:
            return None
        U_new = U_cur[:, remaining]
        V_new = V_cur[:, remaining]
        W_new = W_cur[:, remaining]
        err = verify_decomp(U_new, V_new, W_new, len(remaining))
        if err < 1e-10:
            return (U_new, V_new, W_new)
        return None

    current_U = U_arr.copy()
    current_V = V_arr.copy()
    current_W = W_arr.copy()
    current_rank = rank

    def build_term_matrix(U_mat, V_mat, W_mat, r):
        M = np.zeros((N, r), dtype=float)
        for ri in range(r):
            M[:, ri] = np.einsum('a,b,c->abc', U_mat[:, ri], V_mat[:, ri], W_mat[:, ri]).flatten()
        return M

    # Repeatedly check for removable columns (single or pairs)
    for iteration in range(50):
        if not time_ok():
            break
        if current_rank <= best_rank:
            break

        M = build_term_matrix(current_U, current_V, current_W, current_rank)

        removed_any = False

        # First pass: try single columns
        for r_elim in range(current_rank):
            if not time_ok():
                break
            other_cols = [s for s in range(current_rank) if s != r_elim]
            M_sub = M[:, other_cols]
            rhs = M[:, r_elim]

            try:
                alpha, residuals, rk, sv = np.linalg.lstsq(M_sub, rhs, rcond=1e-10)
                residual = np.max(np.abs(M_sub @ alpha - rhs))
                if residual < 1e-8:
                    result = try_remove_set(current_U, current_V, current_W, [r_elim])
                    if result is not None:
                        current_U, current_V, current_W = result
                        current_rank -= 1
                        removed_any = True
                        if current_rank < best_rank:
                            best_rank = current_rank
                            best_U = current_U.copy()
                            best_V = current_V.copy()
                            best_W = current_W.copy()
                        break
            except np.linalg.LinAlgError:
                continue

        if removed_any:
            continue

        # Second pass: try pairs of columns
        for r1 in range(current_rank):
            if not time_ok():
                break
            for r2 in range(r1 + 1, current_rank):
                if not time_ok():
                    break
                other_cols = [s for s in range(current_rank) if s != r1 and s != r2]
                if len(other_cols) == 0:
                    continue
                M_sub = M[:, other_cols]
                rhs1 = M[:, r1]
                rhs2 = M[:, r2]
                try:
                    alpha1, _, _, _ = np.linalg.lstsq(M_sub, rhs1, rcond=1e-10)
                    residual1 = np.max(np.abs(M_sub @ alpha1 - rhs1))
                    alpha2, _, _, _ = np.linalg.lstsq(M_sub, rhs2, rcond=1e-10)
                    residual2 = np.max(np.abs(M_sub @ alpha2 - rhs2))
                    if residual1 < 1e-8 and residual2 < 1e-8:
                        result = try_remove_set(current_U, current_V, current_W, [r1, r2])
                        if result is not None:
                            current_U, current_V, current_W = result
                            current_rank -= 2
                            removed_any = True
                            if current_rank < best_rank:
                                best_rank = current_rank
                                best_U = current_U.copy()
                                best_V = current_V.copy()
                                best_W = current_W.copy()
                            break
                except np.linalg.LinAlgError:
                    continue
            if removed_any:
                break

        if not removed_any:
            break

    # Final verification and cleanup
    final_err = verify_decomp(best_U, best_V, best_W, best_rank)
    if final_err > 1e-10:
        best_U = U_arr
        best_V = V_arr
        best_W = W_arr
        best_rank = 28

    # High-precision polish: try to round to nearest half-integer and verify
    U_polished = np.round(best_U * 2) / 2.0
    V_polished = np.round(best_V * 2) / 2.0
    W_polished = np.round(best_W * 2) / 2.0
    polish_err = verify_decomp(U_polished, V_polished, W_polished, best_rank)
    if polish_err < 1e-10:
        best_U = U_polished
        best_V = V_polished
        best_W = W_polished

    U_final = to_list(best_U)
    V_final = to_list(best_V)
    W_final = to_list(best_W)

    return {"U": U_final, "V": V_final, "W": W_final}
