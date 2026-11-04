import time
import numpy as np

def construct():
    n, m, p = 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8

    start_time = time.time()
    deadline = start_time + 130.0

    # Build the multiplication tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for l in range(m):
            for k in range(p):
                a = i * m + l
                b = l * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Known rank-28 solution via Strassen on 2x2 blocks
    rank = 28
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        # Strassen's 7 multiplications for 2x2 matrices
        # m1 = (a+d)(e+h)
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1

        # m2 = (c+d)e
        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1

        # m3 = a(f-h)
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1

        # m4 = d(g-e)
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1

        # m5 = (a+b)h
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1

        # m6 = (c-d)(g+h)
        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1

        # m7 = (b-d)(f+g)
        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1

    # Block 1: A1 * B11 -> C11
    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 2, 1, 3]
    add_block(0, X1, Y1, Z1)

    # Block 2: A2 * B21 -> C21
    X2 = [2, 3, 6, 7]
    Y2 = [8, 9, 12, 13]
    Z2 = [0, 2, 1, 3]
    add_block(7, X2, Y2, Z2)

    # Block 3: A1 * B12 -> C12
    X3 = [0, 1, 4, 5]
    Y3 = [2, 3, 6, 7]
    Z3 = [4, 6, 5, 7]
    add_block(14, X3, Y3, Z3)

    # Block 4: A2 * B22 -> C22
    X4 = [2, 3, 6, 7]
    Y4 = [10, 11, 14, 15]
    Z4 = [4, 6, 5, 7]
    add_block(21, X4, Y4, Z4)

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    def verify_decomp(U_mat, V_mat, W_mat, rank_val):
        # Vectorized verification
        S = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
        return np.max(np.abs(S - T))

    # Initial verification
    max_err = verify_decomp(U_arr, V_arr, W_arr, rank)
    if max_err > 1e-10:
        # Fallback: return the list version directly if verification fails (should not happen)
        return {"U": U, "V": V, "W": W}

    best_U = U_arr.copy()
    best_V = V_arr.copy()
    best_W = W_arr.copy()
    best_rank = rank

    current_U = U_arr.copy()
    current_V = V_arr.copy()
    current_W = W_arr.copy()
    current_rank = rank

    # Attempt rank reduction
    # We check if any tensor T_r is in the span of the others.
    # We do this by checking the rank of the matrix formed by flattening the tensors.
    
    for _ in range(current_rank):
        if time.time() > deadline - 10:
            break
            
        if current_rank <= 1:
            break
            
        # Build M where columns are flattened tensors
        M = np.zeros((nm * mp * pn, current_rank), dtype=float)
        for r in range(current_rank):
            M[:, r] = np.einsum('a,b,c->abc', current_U[:, r], current_V[:, r], current_W[:, r]).flatten()
        
        # Check numerical rank
        try:
            # Use SVD to find numerical rank
            U_svd, s, Vt_svd = np.linalg.svd(M, full_matrices=False)
            tol = 1e-9
            num_rank = np.sum(s > tol)
            
            if num_rank < current_rank:
                # Find a column to eliminate
                reduced = False
                # Try to eliminate columns corresponding to small singular values
                # Or just iterate
                for r_elim in range(current_rank):
                    other_cols = [i for i in range(current_rank) if i != r_elim]
                    M_sub = M[:, other_cols]
                    rhs = M[:, r_elim]
                    
                    try:
                        alpha, residuals, rk, sv = np.linalg.lstsq(M_sub, rhs, rcond=1e-10)
                        residual = np.max(np.abs(M_sub @ alpha - rhs))
                        if residual < 1e-8:
                            # Absorb
                            new_U = current_U[:, other_cols].copy()
                            new_V = current_V[:, other_cols].copy()
                            new_W = current_W[:, other_cols].copy()
                            
                            coeffs = 1.0 + alpha
                            
                            for idx, j in enumerate(other_cols):
                                new_U[:, idx] *= coeffs[idx]
                                new_V[:, idx] *= coeffs[idx]
                                new_W[:, idx] *= coeffs[idx]
                                
                            new_rank = current_rank - 1
                            
                            # Verify
                            err = verify_decomp(new_U, new_V, new_W, new_rank)
                            if err < 1e-10:
                                current_U = new_U
                                current_V = new_V
                                current_W = new_W
                                current_rank = new_rank
                                if current_rank < best_rank:
                                    best_rank = current_rank
                                    best_U = current_U.copy()
                                    best_V = current_V.copy()
                                    best_W = current_W.copy()
                                reduced = True
                                break
                    except np.linalg.LinAlgError:
                        continue
                if reduced:
                    continue
        except Exception:
            break

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

    U_final = to_list(best_U)
    V_final = to_list(best_V)
    W_final = to_list(best_W)

    return {"U": U_final, "V": V_final, "W": W_final}
