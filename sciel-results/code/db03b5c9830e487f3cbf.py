import time
import numpy as np

def construct():
    n, m, p = 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8

    start_time = time.time()
    deadline = start_time + 130.0

    # Build the multiplication tensor T of shape (nm, mp, pn)
    # Indexing: A is n x m (rows i, cols l), B is m x p (rows l, cols k), C is p x n (rows k, cols i)
    # T[a, b, c] = 1 if a = i*m+l, b = l*p+k, c = k*n+i
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for l in range(m):
            for k in range(p):
                a = i * m + l
                b = l * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Rank-28 solution via Strassen on 2x2 blocks
    rank = 28
    U = [[0.0] * rank for _ in range(nm)]
    V = [[0.0] * rank for _ in range(mp)]
    W = [[0.0] * rank for _ in range(pn)]

    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        # Strassen's 7 multiplications for 2x2 block
        # X_rows = [x00, x01, x10, x11] indices in U
        # Y_rows = [y00, y01, y10, y11] indices in V
        # Z_rows = [z00, z01, z10, z11] indices in W
        
        # M1 = (A10+A11)(B10+B11)
        U[X_rows[2]][col_offset + 0] += 1.0
        U[X_rows[3]][col_offset + 0] += 1.0
        V[Y_rows[2]][col_offset + 0] += 1.0
        V[Y_rows[3]][col_offset + 0] += 1.0
        W[Z_rows[0]][col_offset + 0] += 1.0
        W[Z_rows[1]][col_offset + 0] += 1.0
        W[Z_rows[2]][col_offset + 0] += 1.0
        W[Z_rows[3]][col_offset + 0] += 1.0

        # M2 = A11(B20-B10)
        U[X_rows[3]][col_offset + 1] += 1.0
        V[Y_rows[2]][col_offset + 1] += 1.0
        V[Y_rows[0]][col_offset + 1] -= 1.0
        W[Z_rows[2]][col_offset + 1] += 1.0
        W[Z_rows[3]][col_offset + 1] += 1.0

        # M3 = (A00-A10)B11
        U[X_rows[0]][col_offset + 2] += 1.0
        U[X_rows[2]][col_offset + 2] -= 1.0
        V[Y_rows[3]][col_offset + 2] += 1.0
        W[Z_rows[0]][col_offset + 2] += 1.0
        W[Z_rows[1]][col_offset + 2] += 1.0

        # M4 = A00(B11-B21)
        U[X_rows[0]][col_offset + 3] += 1.0
        V[Y_rows[3]][col_offset + 3] += 1.0
        V[Y_rows[1]][col_offset + 3] -= 1.0
        W[Z_rows[0]][col_offset + 3] += 1.0

        # M5 = (A00+A11)(B00+B21)
        U[X_rows[0]][col_offset + 4] += 1.0
        U[X_rows[3]][col_offset + 4] += 1.0
        V[Y_rows[0]][col_offset + 4] += 1.0
        V[Y_rows[1]][col_offset + 4] += 1.0
        W[Z_rows[1]][col_offset + 4] += 1.0
        W[Z_rows[3]][col_offset + 4] += 1.0

        # M6 = (A10-A00)(B00+B10)
        U[X_rows[2]][col_offset + 5] += 1.0
        U[X_rows[0]][col_offset + 5] -= 1.0
        V[Y_rows[0]][col_offset + 5] += 1.0
        V[Y_rows[2]][col_offset + 5] += 1.0
        W[Z_rows[2]][col_offset + 5] += 1.0

        # M7 = (A01-A11)(B11+B21)
        U[X_rows[1]][col_offset + 6] += 1.0
        U[X_rows[3]][col_offset + 6] -= 1.0
        V[Y_rows[3]][col_offset + 6] += 1.0
        V[Y_rows[1]][col_offset + 6] += 1.0
        W[Z_rows[1]][col_offset + 6] += 1.0
        W[Z_rows[3]][col_offset + 6] += 1.0

    # Block 1: A rows 0-1, B rows 0-1, C cols 0-1 (i=0,1; l=0,1; k=0,1)
    # X = [A(0,0), A(0,1), A(1,0), A(1,1)] = [0, 1, 4, 5]
    # Y = [B(0,0), B(0,1), B(1,0), B(1,1)] = [0, 1, 4, 5]
    # Z = [C(0,0), C(0,1), C(1,0), C(1,1)] = [0, 2, 4, 6]
    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 2, 4, 6]
    add_block(0, X1, Y1, Z1)

    # Block 2: A rows 0-1, B rows 2-3, C cols 0-1 (i=0,1; l=2,3; k=0,1)
    # X = [A(0,2), A(0,3), A(1,2), A(1,3)] = [2, 3, 6, 7]
    # Y = [B(2,0), B(2,1), B(3,0), B(3,1)] = [8, 9, 12, 13]
    # Z = [C(0,0), C(0,1), C(1,0), C(1,1)] = [0, 2, 4, 6]
    X2 = [2, 3, 6, 7]
    Y2 = [8, 9, 12, 13]
    Z2 = [0, 2, 4, 6]
    add_block(7, X2, Y2, Z2)

    # Block 3: A rows 0-1, B rows 0-1, C cols 2-3 (i=0,1; l=0,1; k=2,3)
    # X = [A(0,0), A(0,1), A(1,0), A(1,1)] = [0, 1, 4, 5]
    # Y = [B(0,2), B(0,3), B(1,2), B(1,3)] = [2, 3, 6, 7]
    # Z = [C(2,0), C(2,1), C(3,0), C(3,1)] = [1, 3, 5, 7]
    X3 = [0, 1, 4, 5]
    Y3 = [2, 3, 6, 7]
    Z3 = [1, 3, 5, 7]
    add_block(14, X3, Y3, Z3)

    # Block 4: A rows 0-1, B rows 2-3, C cols 2-3 (i=0,1; l=2,3; k=2,3)
    # X = [A(0,2), A(0,3), A(1,2), A(1,3)] = [2, 3, 6, 7]
    # Y = [B(2,2), B(2,3), B(3,2), B(3,3)] = [10, 11, 14, 15]
    # Z = [C(2,0), C(2,1), C(3,0), C(3,1)] = [1, 3, 5, 7]
    X4 = [2, 3, 6, 7]
    Y4 = [10, 11, 14, 15]
    Z4 = [1, 3, 5, 7]
    add_block(21, X4, Y4, Z4)

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    def verify_decomp(U_mat, V_mat, W_mat, rank_val):
        S = np.zeros((nm, mp, pn), dtype=float)
        for r in range(rank_val):
            S += np.einsum('a,b,c->abc', U_mat[:, r], V_mat[:, r], W_mat[:, r])
        return np.max(np.abs(S - T))

    max_err = verify_decomp(U_arr, V_arr, W_arr, rank)
    if max_err > 1e-10:
        return {"U": U, "V": V, "W": W}

    # Try to find linear dependencies
    M = np.zeros((nm * mp * pn, rank), dtype=float)
    for r in range(rank):
        M[:, r] = np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r]).flatten()

    best_rank = rank
    best_U = U_arr.copy()
    best_V = V_arr.copy()
    best_W = W_arr.copy()

    # Check for linear dependencies using SVD
    try:
        U_svd, S_svd, Vt_svd = np.linalg.svd(M, full_matrices=False)
        tol = 1e-10
        num_nonzero = np.sum(S_svd > tol)
        
        if num_nonzero < rank:
            for r_elim in range(rank):
                if time.time() > deadline - 10:
                    break
                
                other_cols = [s for s in range(rank) if s != r_elim]
                M_sub = M[:, other_cols]
                rhs = M[:, r_elim]
                
                try:
                    alpha, residuals, rk, sv = np.linalg.lstsq(M_sub, rhs, rcond=tol)
                    residual = np.max(np.abs(M_sub @ alpha - rhs))
                    if residual < 1e-8:
                        new_rank = rank - 1
                        new_U = np.delete(U_arr, r_elim, axis=1)
                        new_V = np.delete(V_arr, r_elim, axis=1)
                        new_W = np.delete(W_arr, r_elim, axis=1)
                        
                        err = verify_decomp(new_U, new_V, new_W, new_rank)
                        if err < 1e-10:
                            best_rank = new_rank
                            best_U = new_U
                            best_V = new_V
                            best_W = new_W
                            rank = new_rank
                            U_arr = new_U
                            V_arr = new_V
                            W_arr = new_W
                            
                            M = np.zeros((nm * mp * pn, rank), dtype=float)
                            for r in range(rank):
                                M[:, r] = np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r]).flatten()
                            break
                except np.linalg.LinAlgError:
                    continue
    except Exception:
        pass

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
