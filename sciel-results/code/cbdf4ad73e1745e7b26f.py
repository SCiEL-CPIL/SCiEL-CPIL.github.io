"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    deadline = time.time() + 130  # safety margin before 150s kill

    def build_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    T = build_tensor()

    def verify(U, V, W, tol=1e-8):
        """Check exact decomposition."""
        recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.max(np.abs(recon - T)) < tol

    def to_lists(U, V, W, r):
        """Convert numpy arrays to list-of-lists format."""
        U_list = [[float(U[i, t]) for t in range(r)] for i in range(n * m)]
        V_list = [[float(V[i, t]) for t in range(r)] for i in range(m * p)]
        W_list = [[float(W[i, t]) for t in range(r)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}

    # === ROW-SHARING DECOMPOSITION ===
    # Key insight: A is 2x4, B is 4x4, C is 2x4
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Split B into left (cols 0,1) and right (cols 2,3):
    #   C[:,0:2] = A[:,:2] * B[0:2,0:2] + A[:,2:4] * B[2:4,0:2]
    #   C[:,2:4] = A[:,:2] * B[0:2,2:4] + A[:,2:4] * B[2:4,2:4]
    #
    # For C[:,0:2] (2x2 result):
    #   Row 0: [A[0,0], A[0,1]] * B_LL + [A[0,2], A[0,3]] * B_BL
    #   Row 1: [A[1,0], A[1,1]] * B_LL + [A[1,2], A[1,3]] * B_BL
    #
    # Couple both rows: M_LL = [A[0,0];A[1,0];A[0,1];A[1,1]] * B_LL
    # This is a (4x2) * (2x2) -> (4x2) problem where the 4x2 is:
    #   [[A[0,0], A[0,1]], [A[1,0], A[1,1]]] reshaped
    # Actually, let's think of it as: for each column j of the 4x2 A-block,
    # multiply by B_LL (2x2) to get a 2-vector, then place into the result.
    #
    # Better: treat the 4x2 * 2x2 -> 4x2 as 2 independent 2x2 * 2x1 products
    # sharing the same B. Rank-4 for this (schoolbook per column).
    # But we can use Strassen-like sharing: rank 7 for 2x2*2x2, but here
    # the "right" factor is 2x1, so rank 3 per column? No.
    #
    # Let's just do the direct rank-4 per 4x2*2x2 block:
    # M_LL_col0: A_col0 (2x1) * B_LL (2x2) -> 2x1, rank 2
    # M_LL_col1: A_col1 (2x1) * B_LL (2x2) -> 2x1, rank 2
    # Total rank 4 for the 4x2 * 2x2 block.
    #
    # 4 such blocks (LL, BL, LR, BR) * rank 4 = rank 16 total!

    def build_row_sharing_decomp():
        """
        Build rank-16 decomposition using row-sharing.
        
        Blocks:
        - LL: A[:,0:2] (2x2) * B[0:2,0:2] (2x2) -> C[:,0:2] (2x2)
        - BL: A[:,2:4] (2x2) * B[2:4,0:2] (2x2) -> C[:,0:2] (2x2)  
        - LR: A[:,0:2] (2x2) * B[0:2,2:4] (2x2) -> C[:,2:4] (2x2)
        - BR: A[:,2:4] (2x2) * B[2:4,2:4] (2x2) -> C[:,2:4] (2x2)
        
        Each 2x2 * 2x2 -> 2x2 uses rank-4 (schoolbook):
          C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
          C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
          C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
          C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        
        But wait - that's rank 4 per block, giving 16 total.
        However, we can do better by noting that within each block,
        the two rows of A are independent but share B.
        
        Actually, for a 2x2 * 2x2 product, the minimal rank is 7 (Strassen).
        Schoolbook is 4. So rank-4 per block is correct for schoolbook.
        
        4 blocks * 4 = 16 total rank.
        """
        rank = 16
        U = np.zeros((8, rank))   # n*m = 8
        V = np.zeros((16, rank))  # m*p = 16
        W = np.zeros((8, rank))   # p*n = 8
        
        # For each block, we have 4 rank-1 terms (schoolbook 2x2*2x2)
        # Block LL: A rows 0,1 cols 0,1; B rows 0,1 cols 0,1; C cols 0,1
        # A global indices: i*4 + j for i in {0,1}, j in {0,1}
        # B global indices: j*4 + k for j in {0,1}, k in {0,1}
        # C global indices: k*2 + i for k in {0,1}, i in {0,1}
        
        blocks = [
            # (a_row_offsets, a_col_offsets, b_row_offsets, b_col_offsets, c_col_offsets)
            ([0, 1], [0, 1], [0, 1], [0, 1], [0, 1]),  # LL
            ([0, 1], [2, 3], [2, 3], [0, 1], [0, 1]),  # BL
            ([0, 1], [0, 1], [0, 1], [2, 3], [2, 3]),  # LR
            ([0, 1], [2, 3], [2, 3], [2, 3], [2, 3]),  # BR
        ]
        
        for blk_idx, (a_rows, a_cols, b_rows, b_cols, c_cols) in enumerate(blocks):
            base_col = blk_idx * 4
            
            # 4 schoolbook terms for 2x2 * 2x2
            for local_i in range(2):  # row of A/C
                for local_k in range(2):  # col of B/C
                    col = base_col + local_i * 2 + local_k
                    
                    # Sum over local_j
                    for local_j in range(2):
                        a_global = a_rows[local_i] * 4 + a_cols[local_j]
                        b_global = b_rows[local_j] * 4 + b_cols[local_k]
                        c_global = c_cols[local_k] * 2 + a_rows[local_i]
                        
                        U[a_global, col] += 1.0
                        V[b_global, col] += 1.0
                        W[c_global, col] += 1.0
        
        return U, V, W, rank
    
    # Build the rank-16 row-sharing decomposition
    U16, V16, W16, rank16 = build_row_sharing_decomp()
    
    if verify(U16, V16, W16, tol=1e-8):
        best_result = to_lists(U16, V16, W16, rank16)
        best_rank = rank16
    else:
        # Fallback to Strassen rank-28
        def strassen_2x2_decomp():
            U = np.zeros((4, 7))
            V = np.zeros((4, 7))
            W = np.zeros((4, 7))
            U[0, 0] = 1.0; U[3, 0] = 1.0
            V[0, 0] = 1.0; V[3, 0] = 1.0
            W[0, 0] = 1.0; W[3, 0] = 1.0
            U[2, 1] = 1.0; U[3, 1] = 1.0
            V[0, 1] = 1.0
            W[1, 1] = 1.0; W[3, 1] = -1.0
            U[0, 2] = 1.0
            V[1, 2] = 1.0; V[3, 2] = -1.0
            W[2, 2] = 1.0; W[3, 2] = 1.0
            U[3, 3] = 1.0
            V[2, 3] = 1.0; V[0, 3] = -1.0
            W[0, 3] = 1.0; W[1, 3] = 1.0
            U[0, 4] = 1.0; U[1, 4] = 1.0
            V[3, 4] = 1.0
            W[0, 4] = -1.0; W[2, 4] = 1.0
            U[2, 5] = 1.0; U[0, 5] = -1.0
            V[0, 5] = 1.0; V[1, 5] = 1.0
            W[3, 5] = 1.0
            U[1, 6] = 1.0; U[3, 6] = -1.0
            V[2, 6] = 1.0; V[3, 6] = 1.0
            W[0, 6] = 1.0
            return U, V, W

        U_s, V_s, W_s = strassen_2x2_decomp()
        full_rank = 28
        U_full = np.zeros((n * m, full_rank))
        V_full = np.zeros((m * p, full_rank))
        W_full = np.zeros((p * n, full_rank))

        blocks = [
            ([0, 1], [0, 1], [0, 1], [0, 1], [0, 1]),
            ([0, 1], [2, 3], [2, 3], [0, 1], [0, 1]),
            ([0, 1], [0, 1], [0, 1], [2, 3], [2, 3]),
            ([0, 1], [2, 3], [2, 3], [2, 3], [2, 3]),
        ]

        for blk_idx, (a_rows, a_cols, b_rows, b_cols, c_cols) in enumerate(blocks):
            for t in range(7):
                col = blk_idx * 7 + t
                for li in range(2):
                    for lj in range(2):
                        local_a = li * 2 + lj
                        global_a = a_rows[li] * 4 + a_cols[lj]
                        U_full[global_a, col] = U_s[local_a, t]
                for lj in range(2):
                    for lk in range(2):
                        local_b = lj * 2 + lk
                        global_b = b_rows[lj] * 4 + b_cols[lk]
                        V_full[global_b, col] = V_s[local_b, t]
                for lk in range(2):
                    for li in range(2):
                        local_c = lk * 2 + li
                        global_c = c_cols[lk] * 2 + a_rows[li]
                        W_full[global_c, col] = W_s[local_c, t]

        if not verify(U_full, V_full, W_full, tol=1e-8):
            rank = n * m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1.0
                        V[j * p + k, col] = 1.0
                        W[k * n + i, col] = 1.0
                        col += 1
            return to_lists(U, V, W, rank)

        best_result = to_lists(U_full, V_full, W_full, full_rank)
        best_rank = full_rank

    # Try to reduce further with optimization if time permits
    if time.time() < deadline - 30:
        try:
            from scipy.optimize import minimize

            def flatten_params(U, V, W):
                return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

            def unflatten_params(params, r):
                size_u = n * m * r
                size_v = m * p * r
                size_w = p * n * r
                U = params[:size_u].reshape(n * m, r)
                V = params[size_u:size_u + size_v].reshape(m * p, r)
                W = params[size_u + size_v:].reshape(p * n, r)
                return U, V, W

            def objective(params, r):
                U, V, W = unflatten_params(params, r)
                recon = np.einsum('at,bt,ct->abc', U, V, W)
                return np.sum((recon - T)**2)

            # Try to reduce below current best
            for target_rank in range(max(1, best_rank - 1), max(0, best_rank - 4)):
                if time.time() >= deadline - 15:
                    break

                for restart in range(3):
                    if time.time() >= deadline - 10:
                        break

                    if restart == 0:
                        U_init = best_result["U"][:n*m] and np.array(best_result["U"])
                        V_init = np.array(best_result["V"])
                        W_init = np.array(best_result["W"])
                        U_init = U_init[:, :target_rank] if U_init.shape[1] >= target_rank else np.pad(U_init, ((0,0),(0,target_rank-U_init.shape[1])))
                        V_init = V_init[:, :target_rank] if V_init.shape[1] >= target_rank else np.pad(V_init, ((0,0),(0,target_rank-V_init.shape[1])))
                        W_init = W_init[:, :target_rank] if W_init.shape[1] >= target_rank else np.pad(W_init, ((0,0),(0,target_rank-W_init.shape[1])))
                    else:
                        U_init = np.random.randn(n * m, target_rank) * 0.5
                        V_init = np.random.randn(m * p, target_rank) * 0.5
                        W_init = np.random.randn(p * n, target_rank) * 0.5

                    init_params = flatten_params(U_init, V_init, W_init)

                    try:
                        res = minimize(
                            objective,
                            init_params,
                            args=(target_rank,),
                            method='L-BFGS-B',
                            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 8000}
                        )

                        if res.fun < 1e-8:
                            U_opt, V_opt, W_opt = unflatten_params(res.x, target_rank)
                            if verify(U_opt, V_opt, W_opt, tol=1e-6):
                                best_rank = target_rank
                                best_result = to_lists(U_opt, V_opt, W_opt, target_rank)
                                break

                    except Exception:
                        pass

        except Exception:
            pass

    return best_result
# EVOLVE-BLOCK-END
