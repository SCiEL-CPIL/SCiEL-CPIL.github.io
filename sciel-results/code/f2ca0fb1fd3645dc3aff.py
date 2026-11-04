import numpy as np
import time
from fractions import Fraction

def construct():
    n, m, p = 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # 1. Construct the base rank-28 Strassen decomposition
    rank = 28
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        # M1
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1

        # M2
        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1

        # M3
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1

        # M4
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1

        # M5
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1

        # M6
        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1

        # M7
        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1

    # Define the 4 blocks of the 2x4 * 4x4 multiplication
    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 2, 1, 3]
    add_block(0, X1, Y1, Z1)

    X2 = [2, 3, 6, 7]
    Y2 = [8, 9, 12, 13]
    Z2 = [0, 2, 1, 3]
    add_block(7, X2, Y2, Z2)

    X3 = [0, 1, 4, 5]
    Y3 = [2, 3, 6, 7]
    Z3 = [4, 6, 5, 7]
    add_block(14, X3, Y3, Z3)

    X4 = [2, 3, 6, 7]
    Y4 = [10, 11, 14, 15]
    Z4 = [4, 6, 5, 7]
    add_block(21, X4, Y4, Z4)

    # 2. Find the true rank of the tensor via exact rational null-space computation
    # The tensor has 28 rank-1 terms. We need to find how many are linearly independent.
    # We form the matrix A of size (8*16*8) x 28 where A[(i,j,k), r] = U[i,r]*V[j,r]*W[k,r]
    # The rank of A is the tensor rank (if the decomposition is minimal).
    # If rank < 28, we can reduce.
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the tensor product matrix
    U_exp = U_np[:, np.newaxis, np.newaxis, :]
    V_exp = V_np[np.newaxis, :, np.newaxis, :]
    W_exp = W_np[np.newaxis, np.newaxis, :, :]
    
    Tensor_Prod = U_exp * V_exp * W_exp
    A_matrix = Tensor_Prod.reshape(-1, 28)
    
    # Use SVD to find the rank
    try:
        U_svd, s, Vt = np.linalg.svd(A_matrix, full_matrices=False)
        tol = 1e-10
        rank_est = int(np.sum(s > tol))
        
        if rank_est < 28:
            # We can reduce the rank to rank_est
            # Use the null space to eliminate redundant terms
            null_vecs = Vt[rank_est:]
            
            # Find a null vector with maximum support to eliminate
            best_c = None
            best_support = 0
            for c in null_vecs:
                support = int(np.sum(np.abs(c) > tol))
                if support > best_support:
                    best_support = support
                    best_c = c
            
            if best_c is not None and best_support > 1:
                max_idx = int(np.argmax(np.abs(best_c)))
                max_val = best_c[max_idx]
                
                if abs(max_val) > tol:
                    new_rank = 27
                    keep_indices = [i for i in range(28) if i != max_idx]
                    
                    new_U = np.zeros((8, new_rank))
                    new_V = np.zeros((16, new_rank))
                    new_W = np.zeros((8, new_rank))
                    
                    for new_col, old_col in enumerate(keep_indices):
                        scale = 1 - best_c[old_col] / max_val
                        new_U[:, new_col] = U_np[:, old_col] * scale
                        new_V[:, new_col] = V_np[:, old_col] * scale
                        new_W[:, new_col] = W_np[:, old_col] * scale
                    
                    # Verify
                    T_old_vec = A_matrix @ np.ones(28)
                    T_new_vec = np.zeros(1024)
                    for new_col, old_col in enumerate(keep_indices):
                        scale = 1 - best_c[old_col] / max_val
                        T_new_vec += scale * A_matrix[:, old_col]
                    
                    if np.allclose(T_old_vec, T_new_vec, atol=1e-8):
                        # Check for integer/half-integer entries
                        if (np.allclose(new_U * 2, np.round(new_U * 2), atol=1e-6) and
                            np.allclose(new_V * 2, np.round(new_V * 2), atol=1e-6) and
                            np.allclose(new_W * 2, np.round(new_W * 2), atol=1e-6)):
                            return {"U": new_U.tolist(), "V": new_V.tolist(), "W": new_W.tolist()}
    except Exception:
        pass

    # Fallback to rank 28
    return {"U": U, "V": V, "W": W}
