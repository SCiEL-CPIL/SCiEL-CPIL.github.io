"""Matrix multiplication 4,4,5 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    nm, mp, pn = n * m, m * p, p * n  # 16, 20, 20
    start_time = time.time()
    time_limit = 140  # Leave margin before 150s kill

    # ------------------------------------------------------------------
    # Helper: verification via einsum
    # ------------------------------------------------------------------
    def verify_numpy(U, V, W, r):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        U3 = U_arr.reshape(n, m, r)
        V3 = V_arr.reshape(m, p, r)
        W3 = W_arr.reshape(p, n, r)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.max(np.abs(T - 1.0))

    # ------------------------------------------------------------------
    # Build the 4x4x4 Strassen decomposition (rank 49)
    # Using the clean recursive Strassen structure from inspiration 1
    # ------------------------------------------------------------------
    def build_strassen_4x4x4():
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))

        # Strassen definitions for 2x2x2
        A_side = [
            [(0, 1.0), (3, 1.0)],
            [(2, 1.0), (3, 1.0)],
            [(0, 1.0)],
            [(3, 1.0)],
            [(0, 1.0), (1, 1.0)],
            [(2, 1.0), (0, -1.0)],
            [(1, 1.0), (3, -1.0)],
        ]
        
        B_side = [
            [(0, 1.0), (3, 1.0)],
            [(0, 1.0)],
            [(1, 1.0), (3, -1.0)],
            [(2, 1.0), (0, -1.0)],
            [(3, 1.0)],
            [(0, 1.0), (1, 1.0)],
            [(2, 1.0), (3, 1.0)],
        ]
        
        out_coeff = [
            {0: 1.0, 3: 1.0},
            {2: 1.0, 3: -1.0},
            {1: 1.0, 3: 1.0},
            {0: 1.0, 2: 1.0},
            {0: -1.0, 1: 1.0},
            {3: 1.0},
            {0: 1.0},
        ]

        # For 4x4x4, we partition A into 4 blocks (bi=0..3), B into 4 blocks, C into 4 blocks.
        # Each block is 2x2.
        # We apply Strassen at level 1 (7 products of 2x2 blocks).
        # Then for each 2x2 block product, we apply Strassen at level 2 (7 scalar products).
        
        for mk in range(7):
            for nk in range(7):
                col = mk * 7 + nk
                
                # Level 1: Combine A blocks
                for (bi, c1) in A_side[mk]:
                    # bi is the index of the 2x2 block of A (0..3)
                    # bi = 2*bi_row + bi_col
                    bi_row = bi // 2
                    bi_col = bi % 2
                    
                    # Level 2: Combine A sub-blocks (1x1)
                    for (bi2, c2) in A_side[nk]:
                        # bi2 is the index of the 1x1 element within the 2x2 block
                        u_local = bi2 // 2
                        v_local = bi2 % 2
                        
                        # Global indices for A
                        i = 2 * bi_row + u_local
                        j = 2 * bi_col + v_local
                        U[i * m + j, col] += c1 * c2

                # Level 1: Combine B blocks
                for (bi, c1) in B_side[mk]:
                    bi_row = bi // 2
                    bi_col = bi % 2
                    
                    # Level 2: Combine B sub-blocks (1x1)
                    for (bi2, c2) in B_side[nk]:
                        u_local = bi2 // 2
                        v_local = bi2 % 2
                        
                        # Global indices for B
                        j = 2 * bi_row + u_local
                        k = 2 * bi_col + v_local
                        V[j * p + k, col] += c1 * c2

                # Level 1: Combine C blocks
                for (bi_outer, c_outer) in out_coeff[mk].items():
                    bi_outer_row = bi_outer // 2
                    bi_outer_col = bi_outer % 2
                    
                    # Level 2: Combine C sub-blocks (1x1)
                    for (bi_inner, c_inner) in out_coeff[nk].items():
                        u_local = bi_inner // 2
                        v_local = bi_inner % 2
                        
                        # Global indices for C
                        i = 2 * bi_outer_row + u_local
                        k = 2 * bi_outer_col + v_local
                        W[k * n + i, col] += c_outer * c_inner

        return U, V, W

    # Build the 4x4x4 Strassen decomposition
    U_str, V_str, W_str = build_strassen_4x4x4()

    # Verify Strassen part
    err_str = verify_numpy(U_str.tolist(), V_str.tolist(), W_str.tolist(), 49)
    if err_str > 1e-6:
        # Fallback: Schoolbook 4x4x4 (rank 64)
        U_str = np.zeros((16, 64))
        V_str = np.zeros((16, 64))
        W_str = np.zeros((16, 64))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_str[i*4+j, col] = 1
                    V_str[j*4+k, col] = 1
                    W_str[k*4+i, col] = 1
                    col += 1
        rank_strassen = 64
    else:
        rank_strassen = 49

    # ------------------------------------------------------------------
    # Build the full 4x4x5 decomposition:
    # Part 1: 4x4x4 (Strassen, rank 49) - uses first 4 columns of B (k=0..3)
    # Part 2: 4x4x1 (schoolbook, rank 16) - uses last column of B (k=4)
    # ------------------------------------------------------------------
    rank_total = rank_strassen + 16
    U = np.zeros((16, rank_total))
    V = np.zeros((20, rank_total))  # m*p = 4*5 = 20
    W = np.zeros((20, rank_total))  # p*n = 5*4 = 20

    # Part 1: Strassen 4x4x4
    U[:, :rank_strassen] = U_str
    V[:16, :rank_strassen] = V_str
    W[:16, :rank_strassen] = W_str

    # Part 2: Schoolbook 4x4x1
    t = 0
    for i in range(4):
        for j in range(4):
            U[i*4+j, rank_strassen+t] = 1
            V[j*4+4, rank_strassen+t] = 1
            W[4*4+i, rank_strassen+t] = 1
            t += 1

    # Verify
    err = verify_numpy(U.tolist(), V.tolist(), W.tolist(), rank_total)
    if err > 1e-6:
        # Fallback to full schoolbook
        U = np.zeros((16, 80))
        V = np.zeros((20, 80))
        W = np.zeros((20, 80))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    U[i*4+j, col] = 1
                    V[j*5+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        rank_total = 80
        err = verify_numpy(U.tolist(), V.tolist(), W.tolist(), rank_total)

    # Convert to lists with clean values
    U_list = [[float(U[i][c]) for c in range(rank_total)] for i in range(16)]
    V_list = [[float(V[i][c]) for c in range(rank_total)] for i in range(20)]
    W_list = [[float(W[i][c]) for c in range(rank_total)] for i in range(20)]

    # Round to nearest 0.5
    U_list = [[round(x * 2) / 2.0 for x in row] for row in U_list]
    V_list = [[round(x * 2) / 2.0 for x in row] for row in V_list]
    W_list = [[round(x * 2) / 2.0 for x in row] for row in W_list]

    # Final verification
    err = verify_numpy(U_list, V_list, W_list, rank_total)
    if err > 1e-6:
        # Fall back to schoolbook
        U_list = [[0.0] * 80 for _ in range(16)]
        V_list = [[0.0] * 80 for _ in range(20)]
        W_list = [[0.0] * 80 for _ in range(20)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    U_list[i*4+j][col] = 1.0
                    V_list[j*5+k][col] = 1.0
                    W_list[k*4+i][col] = 1.0
                    col += 1
        rank_total = 80

    # ------------------------------------------------------------------
    # Try to reduce rank using optimization (if time allows)
    # ------------------------------------------------------------------
    if time.time() - start_time < time_limit - 10 and rank_total > 55:
        try:
            from scipy.optimize import minimize
        except ImportError:
            minimize = None

        if minimize is not None:
            # Try to find a lower rank decomposition
            for target_rank in range(55, min(rank_total, 65)):
                if time.time() - start_time > time_limit - 5:
                    break

                # Initialize with the current decomposition, zeroing out extra columns
                U_init = np.zeros((16, target_rank))
                V_init = np.zeros((20, target_rank))
                W_init = np.zeros((20, target_rank))
                min_cols = min(rank_total, target_rank)
                U_init[:, :min_cols] = np.array(U_list)[:, :min_cols]
                V_init[:, :min_cols] = np.array(V_list)[:, :min_cols]
                W_init[:, :min_cols] = np.array(W_list)[:, :min_cols]

                x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])

                def objective(x):
                    if time.time() - start_time > time_limit - 3:
                        return 1e10
                    U_flat = x[:16*target_rank]
                    V_flat = x[16*target_rank:16*target_rank+20*target_rank]
                    W_flat = x[16*target_rank+20*target_rank:]

                    U_mat = U_flat.reshape(16, target_rank)
                    V_mat = V_flat.reshape(20, target_rank)
                    W_mat = W_flat.reshape(20, target_rank)

                    U3 = U_mat.reshape(n, m, target_rank)
                    V3 = V_mat.reshape(m, p, target_rank)
                    W3 = W_mat.reshape(p, n, target_rank)

                    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
                    return np.sum((T - 1.0) ** 2)

                try:
                    res = minimize(objective, x0, method='L-BFGS-B',
                                  options={'maxiter': 200, 'maxfun': 2000})

                    if res.fun < 1e-8:
                        x_opt = res.x
                        U_opt = x_opt[:16*target_rank].reshape(16, target_rank)
                        V_opt = x_opt[16*target_rank:16*target_rank+20*target_rank].reshape(20, target_rank)
                        W_opt = x_opt[16*target_rank+20*target_rank:].reshape(20, target_rank)

                        # Round to nearest 0.5
                        U_r = [[round(U_opt[i][c] * 2) / 2.0 for c in range(target_rank)] for i in range(16)]
                        V_r = [[round(V_opt[i][c] * 2) / 2.0 for c in range(target_rank)] for i in range(20)]
                        W_r = [[round(W_opt[i][c] * 2) / 2.0 for c in range(target_rank)] for i in range(20)]

                        err_r = verify_numpy(U_r, V_r, W_r, target_rank)
                        if err_r < 1e-6:
                            U_list = U_r
                            V_list = V_r
                            W_list = W_r
                            rank_total = target_rank
                            break  # Found a valid lower rank, stop searching
                except Exception:
                    pass

    return {"U": U_list, "V": V_list, "W": W_list}
