import time
import numpy as np

def construct():
    n, m, p = 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8

    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Build the multiplication tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for l in range(m):
            for k in range(p):
                a = i * m + l
                b = l * p + k
                c = k * n + i
                T[a, b, c] = 1.0

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
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1
        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1
        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1
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

    def run_als(U_init, V_init, W_init, rank_val, max_iters=50):
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        for _ in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Update U
            T_reshaped = T.reshape(nm, mp * pn)
            M = np.zeros((mp * pn, rank_val), dtype=float)
            for r in range(rank_val):
                M[:, r] = (V[:, r] * W[:, r]).flatten()
            U_new, _, _, _ = np.linalg.lstsq(M.T, T_reshaped, rcond=None)
            U = U_new.T
            
            # Update V
            T_reshaped_v = T.reshape(mp, nm * pn)
            M_v = np.zeros((nm * pn, rank_val), dtype=float)
            for r in range(rank_val):
                M_v[:, r] = (U[:, r] * W[:, r]).flatten()
            V_new, _, _, _ = np.linalg.lstsq(M_v.T, T_reshaped_v, rcond=None)
            V = V_new.T
            
            # Update W
            T_reshaped_w = T.reshape(pn, nm * mp)
            M_w = np.zeros((nm * mp, rank_val), dtype=float)
            for r in range(rank_val):
                M_w[:, r] = (U[:, r] * V[:, r]).flatten()
            W_new, _, _, _ = np.linalg.lstsq(M_w.T, T_reshaped_w, rcond=None)
            W = W_new.T
            
        return U, V, W

    def try_find_solution(target_rank, num_restarts=10):
        for i in range(num_restarts):
            if time.time() > deadline - 5:
                return None
            
            # Warm start: use random subset of columns from known solution
            # Or random init
            if i % 2 == 0:
                # Random init
                U_init = np.random.randn(nm, target_rank) * 0.1
                V_init = np.random.randn(mp, target_rank) * 0.1
                W_init = np.random.randn(pn, target_rank) * 0.1
            else:
                # Warm start from rank 28 solution
                # Pick random target_rank columns
                cols = np.random.choice(rank, target_rank, replace=False)
                U_init = U_arr[:, cols].copy()
                V_init = V_arr[:, cols].copy()
                W_init = W_arr[:, cols].copy()
            
            U_sol, V_sol, W_sol = run_als(U_init, V_init, W_init, target_rank, max_iters=50)
            
            # Round to nearest 0.5
            U_rounded = np.round(U_sol * 2) / 2.0
            V_rounded = np.round(V_sol * 2) / 2.0
            W_rounded = np.round(W_sol * 2) / 2.0
            
            err = verify_decomp(U_rounded, V_rounded, W_rounded, target_rank)
            if err < 1e-10:
                return (U_rounded, V_rounded, W_rounded)
                
            # Also try rounding to integer
            U_int = np.round(U_sol)
            V_int = np.round(V_sol)
            W_int = np.round(W_sol)
            err_int = verify_decomp(U_int, V_int, W_int, target_rank)
            if err_int < 1e-10:
                return (U_int, V_int, W_int)
        
        return None

    # Try to find lower rank solutions
    for target_rank in [26, 27]:
        if time.time() > deadline - 20:
            break
        sol = try_find_solution(target_rank, num_restarts=10)
        if sol is not None:
            best_U, best_V, best_W = sol
            best_rank = target_rank
            break

    # Final verification and cleanup
    final_err = verify_decomp(best_U, best_V, best_W, best_rank)
    if final_err > 1e-10:
        best_U = U_arr
        best_V = V_arr
        best_W = W_arr
        best_rank = 28

    U_final = to_list(best_U)
    V_final = to_list(best_V)
    W_final = to_list(best_W)

    return {"U": U_final, "V": V_final, "W": W_final}
