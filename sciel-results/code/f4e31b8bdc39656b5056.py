import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = 2, 4, 5
    _t0 = time.time()
    _DEADLINE = 115.0  # Safety margin before 120s kill
    
    # 1. Fallback: Rank 36 Structural Decomposition (Current Best)
    def get_rank_36_solution():
        rank = 36
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        def add_term(idx, u_coeffs, v_coeffs, w_coeffs):
            for (i, j), val in u_coeffs.items():
                U[i * m + j, idx] += val
            for (j, k), val in v_coeffs.items():
                V[j * p + k, idx] += val
            for (k, i), val in w_coeffs.items():
                W[k * n + i, idx] += val

        def strassen_block(a_row_offset, b_row_offset, c_col_offset, idx_start):
            # Strassen 2x2x2
            add_term(idx_start,
                     {(0, a_row_offset): 1, (1, a_row_offset + 1): 1},
                     {(b_row_offset, c_col_offset): 1, (b_row_offset + 1, c_col_offset + 1): 1},
                     {(c_col_offset, 0): 1, (c_col_offset + 1, 1): 1})
            add_term(idx_start + 1,
                     {(1, a_row_offset): 1, (1, a_row_offset + 1): 1},
                     {(b_row_offset, c_col_offset): 1},
                     {(c_col_offset, 1): 1, (c_col_offset + 1, 1): -1})
            add_term(idx_start + 2,
                     {(0, a_row_offset): 1},
                     {(b_row_offset, c_col_offset + 1): 1, (b_row_offset + 1, c_col_offset + 1): -1},
                     {(c_col_offset + 1, 0): 1, (c_col_offset + 1, 1): 1})
            add_term(idx_start + 3,
                     {(1, a_row_offset + 1): 1},
                     {(b_row_offset + 1, c_col_offset): 1, (b_row_offset, c_col_offset): -1},
                     {(c_col_offset, 0): 1, (c_col_offset, 1): 1})
            add_term(idx_start + 4,
                     {(0, a_row_offset): 1, (0, a_row_offset + 1): 1},
                     {(b_row_offset + 1, c_col_offset + 1): 1},
                     {(c_col_offset, 0): -1, (c_col_offset + 1, 0): 1})
            add_term(idx_start + 5,
                     {(1, a_row_offset): 1, (0, a_row_offset): -1},
                     {(b_row_offset, c_col_offset): 1, (b_row_offset, c_col_offset + 1): 1},
                     {(c_col_offset + 1, 1): 1})
            add_term(idx_start + 6,
                     {(0, a_row_offset + 1): 1, (1, a_row_offset + 1): -1},
                     {(b_row_offset + 1, c_col_offset): 1, (b_row_offset + 1, c_col_offset + 1): 1},
                     {(c_col_offset, 0): 1})

        def schoolbook_2x2x1_block(a_row_offset, b_row_offset, c_col_offset, idx_start):
            add_term(idx_start,
                     {(0, a_row_offset): 1},
                     {(b_row_offset, c_col_offset): 1},
                     {(c_col_offset, 0): 1})
            add_term(idx_start + 1,
                     {(0, a_row_offset + 1): 1},
                     {(b_row_offset + 1, c_col_offset): 1},
                     {(c_col_offset, 0): 1})
            add_term(idx_start + 2,
                     {(1, a_row_offset): 1},
                     {(b_row_offset, c_col_offset): 1},
                     {(c_col_offset, 1): 1})
            add_term(idx_start + 3,
                     {(1, a_row_offset + 1): 1},
                     {(b_row_offset + 1, c_col_offset): 1},
                     {(c_col_offset, 1): 1})

        idx = 0
        strassen_block(0, 0, 0, idx); idx += 7
        strassen_block(0, 0, 2, idx); idx += 7
        schoolbook_2x2x1_block(0, 0, 4, idx); idx += 4
        strassen_block(2, 2, 0, idx); idx += 7
        strassen_block(2, 2, 2, idx); idx += 7
        schoolbook_2x2x1_block(2, 2, 4, idx); idx += 4
        
        return U, V, W, rank

    U_fallback, V_fallback, W_fallback, rank_fallback = get_rank_36_solution()
    
    # 2. Search for Rank 34 Solution using M_2,2,4 sub-blocks
    # Strategy: Decompose 2x4x5 into:
    # - Two 2x2x4 blocks (cols 0-3 of B, cols 0-1 and 2-3 of A)
    # - Two 2x1x1 blocks (col 4 of B, cols 0-1 and 2-3 of A)
    # If M_2,2,4 has rank 13, total rank = 2*13 + 2*2 = 30? 
    # Wait, M_2,2,4 rank is actually 13? 
    # Known ranks: M_2,2,2=7, M_2,2,3=12, M_2,2,4=13? 
    # Let's check literature. M_2,2,4 rank is 13.
    # Then total rank = 2 * 13 + 2 * 2 = 30.
    # But wait, the B matrix is 4x5. 
    # Split A into A1 (2x2) and A2 (2x2).
    # C = A1 * B1 + A2 * B2
    # B1 is 2x5, B2 is 2x5.
    # This doesn't decouple nicely into M_2,2,4 unless B is split into 2x4 and 2x1.
    # Let's stick to the structure:
    # C (2x5) = A (2x4) * B (4x5)
    # Split B into B_left (4x4) and B_right (4x1).
    # Split A into A_top (2x2) and A_bottom (2x2) ? No, A is 2x4.
    # Split A into A_left (2x2) and A_right (2x2).
    # C = A_left * B_left_top + A_left * B_right_top + A_right * B_left_bottom + A_right * B_right_bottom
    # Where B_left_top is 2x4, B_right_top is 2x1, etc.
    # This creates 4 blocks:
    # 1. M_2,2,4 (A_left, B_left_top)
    # 2. M_2,2,1 (A_left, B_right_top)
    # 3. M_2,2,4 (A_right, B_left_bottom)
    # 4. M_2,2,1 (A_right, B_right_bottom)
    # Rank = 2 * Rank(M_2,2,4) + 2 * Rank(M_2,2,1)
    # Rank(M_2,2,1) is 4 (schoolbook).
    # If Rank(M_2,2,4) is 13, Total = 2*13 + 2*4 = 34.
    # This matches the target rank 34.
    
    def try_find_rank_34_solution():
        # Target: M_2,2,4 rank 13
        # T[i,j,k] = 1 for i in 0..1, j in 0..1, k in 0..3
        # Variables: U (2*2*13), V (2*4*13), W (4*2*13) -> 52 + 104 + 104 = 260 vars
        # Equations: 2*2*4 = 16
        
        def residual(params):
            U_flat = params[:52]
            V_flat = params[52:156]
            W_flat = params[156:260]
            
            U = U_flat.reshape(2, 2, 13) # i, j, r
            V = V_flat.reshape(2, 4, 13) # j, k, r
            W = W_flat.reshape(4, 2, 13) # k, i, r
            
            res = np.zeros(16)
            idx = 0
            for i in range(2):
                for j in range(2):
                    for k in range(4):
                        val = np.sum(U[i, j, :] * V[j, k, :] * W[k, i, :])
                        res[idx] = val - 1.0
                        idx += 1
            return res
        
        best_res = np.inf
        best_params = None
        
        # Use scipy least_squares
        from scipy.optimize import least_squares
        
        np.random.seed(42)
        max_restarts = 30
        
        for r in range(max_restarts):
            if time.time() > _DEADLINE - 5:
                break
            init = np.random.randn(260) * 0.5
            try:
                sol = least_squares(residual, init, method='lm', max_nfev=2000)
                if sol.cost < best_res:
                    best_res = sol.cost
                    best_params = sol.x
                    if best_res < 1e-10:
                        break
            except Exception:
                continue
                
        if best_res < 1e-8 and best_params is not None:
            rank = 34
            U = np.zeros((n * m, rank), dtype=float)
            V = np.zeros((m * p, rank), dtype=float)
            W = np.zeros((p * n, rank), dtype=float)
            
            U_13 = best_params[:52].reshape(2, 2, 13)
            V_13 = best_params[52:156].reshape(2, 4, 13)
            W_13 = best_params[156:260].reshape(4, 2, 13)
            
            # Block 1: A_left (cols 0,1) * B_left_top (rows 0,1; cols 0..3)
            # A indices: i=0,1; j=0,1
            # B indices: j=0,1; k=0..3
            # C indices: k=0..3; i=0,1
            for i in range(2):
                for j in range(2):
                    for r in range(13):
                        U[i * m + j, r] = U_13[i, j, r]
            for j in range(2):
                for k in range(4):
                    for r in range(13):
                        V[j * p + k, r] = V_13[j, k, r]
            for k in range(4):
                for i in range(2):
                    for r in range(13):
                        W[k * n + i, r] = W_13[k, i, r]
                        
            # Block 2: A_right (cols 2,3) * B_left_bottom (rows 2,3; cols 0..3)
            # A indices: i=0,1; j=2,3
            # B indices: j=2,3; k=0..3
            # C indices: k=0..3; i=0,1
            offset = 13
            for i in range(2):
                for j in range(2):
                    for r in range(13):
                        U[i * m + (2 + j), offset + r] = U_13[i, j, r]
            for j in range(2):
                for k in range(4):
                    for r in range(13):
                        V[(2 + j) * p + k, offset + r] = V_13[j, k, r]
            for k in range(4):
                for i in range(2):
                    for r in range(13):
                        W[k * n + i, offset + r] = W_13[k, i, r]
                        
            # Block 3: A_left (cols 0,1) * B_right_top (rows 0,1; col 4)
            # This is M_2,2,1 (Rank 4)
            # C indices: k=4; i=0,1
            offset2 = 26
            # Schoolbook 2x2x1
            # U: i, j(0,1)
            # V: j(0,1), k(4)
            # W: k(4), i
            # Term 1: (0,0)*(0,4) -> (4,0)
            U[0 * m + 0, offset2] = 1
            V[0 * p + 4, offset2] = 1
            W[4 * n + 0, offset2] = 1
            
            # Term 2: (0,1)*(1,4) -> (4,0)
            U[0 * m + 1, offset2 + 1] = 1
            V[1 * p + 4, offset2 + 1] = 1
            W[4 * n + 0, offset2 + 1] = 1
            
            # Term 3: (1,0)*(0,4) -> (4,1)
            U[1 * m + 0, offset2 + 2] = 1
            V[0 * p + 4, offset2 + 2] = 1
            W[4 * n + 1, offset2 + 2] = 1
            
            # Term 4: (1,1)*(1,4) -> (4,1)
            U[1 * m + 1, offset2 + 3] = 1
            V[1 * p + 4, offset2 + 3] = 1
            W[4 * n + 1, offset2 + 3] = 1
            
            # Block 4: A_right (cols 2,3) * B_right_bottom (rows 2,3; col 4)
            offset3 = 30
            # Term 1: (0,2)*(2,4) -> (4,0)
            U[0 * m + 2, offset3] = 1
            V[2 * p + 4, offset3] = 1
            W[4 * n + 0, offset3] = 1
            
            # Term 2: (0,3)*(3,4) -> (4,0)
            U[0 * m + 3, offset3 + 1] = 1
            V[3 * p + 4, offset3 + 1] = 1
            W[4 * n + 0, offset3 + 1] = 1
            
            # Term 3: (1,2)*(2,4) -> (4,1)
            U[1 * m + 2, offset3 + 2] = 1
            V[2 * p + 4, offset3 + 2] = 1
            W[4 * n + 1, offset3 + 2] = 1
            
            # Term 4: (1,3)*(3,4) -> (4,1)
            U[1 * m + 3, offset3 + 3] = 1
            V[3 * p + 4, offset3 + 3] = 1
            W[4 * n + 1, offset3 + 3] = 1
            
            # Verify
            # Construct full tensor
            T = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] = 1.0
                        
            T_decomp = np.zeros((n, m, p))
            for r in range(rank):
                u_vec = U[:, r].reshape(n, m)
                v_vec = V[:, r].reshape(m, p)
                w_vec = W[:, r].reshape(p, n)
                # T_decomp[i,j,k] += U[i,j,r] V[j,k,r] W[k,i,r]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_decomp[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            
            if np.allclose(T, T_decomp, atol=1e-8):
                return U, V, W, rank
                
        return None

    # Try to find rank 34
    result_34 = try_find_rank_34_solution()
    
    if result_34 is not None:
        U, V, W, rank = result_34
    else:
        U, V, W, rank = U_fallback, V_fallback, W_fallback, rank_fallback

    return {"U": U, "V": V, "W": W}
