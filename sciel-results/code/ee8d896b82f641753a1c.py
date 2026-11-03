import numpy as np
import time
import sys

_DEADLINE = time.time() + 110  # Safety margin before 120s limit

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Total rank: 49 (Strassen 4x4x4) + 44 (4 * Rank(2x2x3) = 4 * 11)
    # 2x2x3: 7 Strassen (cols 0,1) + 4 schoolbook (col 2)
    total_rank = 49 + 44
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    current_rank = 0
    
    # --- Part 1: 4x4x4 using Strassen (Rank 49) ---
    # Same as before
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}),       # M2
        ({0: 1}, {1: 1, 3: -1}),      # M3
        ({3: 1}, {2: 1, 0: -1}),      # M4
        ({0: 1, 1: 1}, {3: 1}),       # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}),# M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}) # M7
    ]
    
    sub_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # p0
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),            # p1
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),           # p2
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),           # p3
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),            # p4
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}), # p5
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1})  # p6
    ]
    
    for m_idx in range(7):
        x_def, y_def = strassen_terms[m_idx]
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub = sub_terms[t_idx]
            
            u_coeffs = [0.0] * (n*m)
            v_coeffs = [0.0] * (m*p)
            
            for (li, lj), c_x in x_sub.items():
                for b, coeff_block in x_def.items():
                    bi = b // 2
                    bj = b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    u_idx = i * m + j
                    u_coeffs[u_idx] += c_x * coeff_block
            
            for (li, lj), c_y in y_sub.items():
                for b, coeff_block in y_def.items():
                    bi = b // 2
                    bj = b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    v_idx = i * p + j
                    v_coeffs[v_idx] += c_y * coeff_block
            
            for i in range(n*m):
                U[i][r] = u_coeffs[i]
            for j in range(m*p):
                V[j][r] = v_coeffs[j]
            
            w_coeffs = [0.0] * (p*n)
            
            cm_coeffs = {
                (0,0): 0.0, (0,1): 0.0, (1,0): 0.0, (1,1): 0.0
            }
            
            if t_idx == 0:
                cm_coeffs[(0,0)] += 1.0
                cm_coeffs[(1,1)] += 1.0
            elif t_idx == 1:
                cm_coeffs[(1,0)] += 1.0
                cm_coeffs[(1,1)] += -1.0
            elif t_idx == 2:
                cm_coeffs[(0,1)] += 1.0
                cm_coeffs[(1,1)] += 1.0
            elif t_idx == 3:
                cm_coeffs[(0,0)] += 1.0
                cm_coeffs[(1,0)] += 1.0
            elif t_idx == 4:
                cm_coeffs[(0,0)] += -1.0
                cm_coeffs[(0,1)] += 1.0
            elif t_idx == 5:
                cm_coeffs[(1,1)] += 1.0
            elif t_idx == 6:
                cm_coeffs[(0,0)] += 1.0
            
            c_block_coeffs = {}
            if m_idx == 0:
                c_block_coeffs = {(0,0): 1.0, (1,1): 1.0}
            elif m_idx == 1:
                c_block_coeffs = {(1,0): 1.0, (1,1): -1.0}
            elif m_idx == 2:
                c_block_coeffs = {(0,1): 1.0, (1,1): 1.0}
            elif m_idx == 3:
                c_block_coeffs = {(0,0): 1.0, (1,0): 1.0}
            elif m_idx == 4:
                c_block_coeffs = {(0,0): -1.0, (0,1): 1.0}
            elif m_idx == 5:
                c_block_coeffs = {(1,1): 1.0}
            elif m_idx == 6:
                c_block_coeffs = {(0,0): 1.0}
            
            for (li, lj), cm_c in cm_coeffs.items():
                if cm_c == 0.0: continue
                for (br, bc), cb_c in c_block_coeffs.items():
                    if cb_c == 0.0: continue
                    i_final = 2 * br + li
                    k_final = 2 * bc + lj
                    w_idx = k_final * n + i_final
                    w_coeffs[w_idx] += cm_c * cb_c
            
            for i in range(p*n):
                W[i][r] = w_coeffs[i]

    # --- Part 2: 4x4x3 using 4 * (2x2x3) (Rank 44) ---
    # Cols 4, 5, 6 of B.
    # B_top: rows 0-1, cols 4-6.
    # B_bot: rows 2-3, cols 4-6.
    # A_00: rows 0-1, cols 0-1.
    # A_01: rows 0-1, cols 2-3.
    # A_10: rows 2-3, cols 0-1.
    # A_11: rows 2-3, cols 2-3.
    
    # Blocks of A: (row_block, col_block)
    # 0: (0,0) -> A_00
    # 1: (0,1) -> A_01
    # 2: (1,0) -> A_10
    # 3: (1,1) -> A_11
    
    # For each A block, we multiply with the corresponding B block.
    # A_00 * B_top -> contributes to C_top (rows 0-1)
    # A_01 * B_bot -> contributes to C_top (rows 0-1)
    # A_10 * B_top -> contributes to C_bot (rows 2-3)
    # A_11 * B_bot -> contributes to C_bot (rows 2-3)
    
    def add_2x2x3(a_row_block, a_col_block, b_row_block, c_row_block):
        nonlocal current_rank
        
        # a_row_block: 0 or 1 (row offset 0 or 2)
        # a_col_block: 0 or 1 (col offset 0 or 2)
        # b_row_block: 0 or 1 (row offset 0 or 2)
        # c_row_block: 0 or 1 (row offset 0 or 2)
        
        # B cols are 4, 5, 6.
        # B index: b_row * 7 + b_col
        
        # 11 terms: 7 Strassen (for cols 4,5) + 4 Schoolbook (for col 6)
        
        # Strassen part (cols 4,5)
        # B_00: b_row, col 4
        # B_01: b_row, col 5
        # B_10: b_row+1, col 4
        # B_11: b_row+1, col 5
        
        # A_00: a_row, a_col
        # A_01: a_row, a_col+1
        # A_10: a_row+1, a_col
        # A_11: a_row+1, a_col+1
        
        strassen_2x2x2_terms = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # M1
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}),            # M2
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}),           # M3
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}),           # M4
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}),            # M5
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}), # M6
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1})  # M7
        ]
        
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub = strassen_2x2x2_terms[t_idx]
            
            u_coeffs = [0.0] * (n*m)
            v_coeffs = [0.0] * (m*p)
            
            # A entries
            for (li, lj), c_x in x_sub.items():
                i = 2 * a_row_block + li
                j = 2 * a_col_block + lj
                u_idx = i * m + j
                u_coeffs[u_idx] += c_x
            
            # B entries (cols 4, 5)
            for (li, lj), c_y in y_sub.items():
                i = 2 * b_row_block + li
                j = 4 + lj # Col 4 or 5
                v_idx = i * p + j
                v_coeffs[v_idx] += c_y
            
            for i in range(n*m):
                U[i][r] = u_coeffs[i]
            for j in range(m*p):
                V[j][r] = v_coeffs[j]
            
            w_coeffs = [0.0] * (p*n)
            
            # C entries (cols 4, 5)
            # C_00 (col 4) = M1 + M4 - M5 + M7
            # C_01 (col 5) = M3 + M5
            # C_10 (col 4) = M2 + M4
            # C_11 (col 5) = M1 - M2 + M3 + M6
            
            cm_coeffs = {
                (0,0): 0.0, (0,1): 0.0, (1,0): 0.0, (1,1): 0.0
            }
            
            if t_idx == 0:
                cm_coeffs[(0,0)] += 1.0
                cm_coeffs[(1,1)] += 1.0
            elif t_idx == 1:
                cm_coeffs[(1,0)] += 1.0
                cm_coeffs[(1,1)] += -1.0
            elif t_idx == 2:
                cm_coeffs[(0,1)] += 1.0
                cm_coeffs[(1,1)] += 1.0
            elif t_idx == 3:
                cm_coeffs[(0,0)] += 1.0
                cm_coeffs[(1,0)] += 1.0
            elif t_idx == 4:
                cm_coeffs[(0,0)] += -1.0
                cm_coeffs[(0,1)] += 1.0
            elif t_idx == 5:
                cm_coeffs[(1,1)] += 1.0
            elif t_idx == 6:
                cm_coeffs[(0,0)] += 1.0
            
            for (li, lj), cm_c in cm_coeffs.items():
                if cm_c == 0.0: continue
                i_final = 2 * c_row_block + li
                k_final = 4 + lj # Col 4 or 5
                w_idx = k_final * n + i_final
                w_coeffs[w_idx] += cm_c
            
            for i in range(p*n):
                W[i][r] = w_coeffs[i]
        
        # Schoolbook part (col 6) — 4 terms
        # C[0,6] = A[0,0]B[0,6] + A[0,1]B[1,6]
        # C[1,6] = A[1,0]B[0,6] + A[1,1]B[1,6]

        schoolbook_terms = [
            # (A_pos, B_pos, C_row_in_block)
            ((0,0), (0,0), 0),  # A[0,0]*B[0,6] -> C[0,6]
            ((0,1), (1,0), 0),  # A[0,1]*B[1,6] -> C[0,6]
            ((1,0), (0,0), 1),  # A[1,0]*B[0,6] -> C[1,6]
            ((1,1), (1,0), 1),  # A[1,1]*B[1,6] -> C[1,6]
        ]

        for a_pos, b_pos, c_row in schoolbook_terms:
            r = current_rank
            current_rank += 1

            u_coeffs = [0.0] * (n*m)
            v_coeffs = [0.0] * (m*p)

            i_a = 2 * a_row_block + a_pos[0]
            j_a = 2 * a_col_block + a_pos[1]
            u_coeffs[i_a * m + j_a] = 1.0

            i_b = 2 * b_row_block + b_pos[0]
            j_b = 6
            v_coeffs[i_b * p + j_b] = 1.0

            for i in range(n*m):
                U[i][r] = u_coeffs[i]
            for j in range(m*p):
                V[j][r] = v_coeffs[j]

            w_coeffs = [0.0] * (p*n)
            i_final = 2 * c_row_block + c_row
            k_final = 6
            w_idx = k_final * n + i_final
            w_coeffs[w_idx] = 1.0

            for i in range(p*n):
                W[i][r] = w_coeffs[i]

    # Call for the 4 blocks
    add_2x2x3(0, 0, 0, 0) # A00 * B_top -> C_top
    add_2x2x3(0, 1, 1, 0) # A01 * B_bot -> C_top
    add_2x2x3(1, 0, 0, 1) # A10 * B_top -> C_bot
    add_2x2x3(1, 1, 1, 1) # A11 * B_bot -> C_bot

    # Trim
    U = [row[:current_rank] for row in U]
    V = [row[:current_rank] for row in V]
    W = [row[:current_rank] for row in W]

    # Verify correctness
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Reconstruct tensor: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # Target: T[i,j,k] = 1 if C[i,k] = sum_j A[i,j]*B[j,k], else 0
    # For matrix mult: T[i,j,k] = delta[i,j,k] where the tensor is the bilinear form
    # Actually the tensor M[i,j,k] = 1 if (i,j,k) corresponds to a valid term
    # The matrix multiplication tensor has M[i,j,k]=1 for all valid (i,j,k) in the standard basis
    # More precisely: sum_j A[i,j] B[j,k] = C[i,k]
    # Tensor: T = sum_{i,j,k} e_i^A ⊗ e_j^B ⊗ e_k^C where the coefficient is 1 for the standard basis
    # The decomposition: U[i,r]*V[j,r]*W[k,r] summed over r should equal delta for each (i,j,k)
    # Actually: C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor is: for each (i,j,k), the coefficient in the trilinear form is 1 if it's a valid term
    # The matrix multiplication tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Wait no. The tensor for matrix multiplication A(n×m) * B(m×p) = C(n×p) is:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ f_j ⊗ g_k
    # where the product gives C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 for ALL valid (i,j,k) indices!
    # No wait, that's not right either. Let me think again.
    # The matrix multiplication map is: (A,B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # As a trilinear form or tensor: M = sum_{i,j,k} c_{ijk} e_i^A ⊗ f_j^B ⊗ g_k^C
    # where c_{ijk} = 1 for all i,j,k (since every A[i,j] and B[j,k] pair contributes to exactly one C[i,k])
    # Actually the tensor is: M = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ f_j ⊗ g_k
    # No! The correct statement is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T such that <T, A⊗B⊗C> = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[i,k]
    # We want this to equal sum_{i,k} C[i,k] * (sum_j A[i,j] B[j,k]) ... no.
    # 
    # Let me use the standard definition. The matrix multiplication tensor M_{n,m,p} is defined as:
    # M = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ f_j ⊗ g_k
    # where e_i are basis for A-space (n*m dims), f_j for B-space (m*p dims), g_k for C-space (n*p dims)
    # Wait, the dimensions are: A has n*m entries, B has m*p entries, C has n*p entries.
    # The tensor lives in (R^{nm})* ⊗ (R^{mp})* ⊗ (R^{np})*
    # M = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_{i,j}^A ⊗ f_{j,k}^B ⊗ g_{i,k}^C
    # where e_{i,j}^A is the basis vector for A[i,j], f_{j,k}^B for B[j,k], g_{i,k}^C for C[i,k]
    # So M[i,j,k] (with i indexing A, j indexing B, k indexing C in their respective flattened forms)
    # A index: i_A = i*m + j
    # B index: i_B = j*p + k  
    # C index: i_C = k*n + i
    # M[i_A, i_B, i_C] = 1 if i_A = i*m+j, i_B = j*p+k, i_C = k*n+i for some i,j,k
    # Otherwise 0.
    
    # Verify: reconstruct and check
    if time.time() < _DEADLINE:
        # Build the target tensor
        target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    target[a_idx, b_idx, c_idx] = 1.0
        
        # Reconstructed: sum_r U[:,r] V[:,r] W[:,r] as outer products
        recon = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
        
        max_err = np.max(np.abs(recon - target))
        if max_err > 1e-10:
            # Fallback: return schoolbook
            rank_sb = n * m * p
            U_sb = np.zeros((n*m, rank_sb))
            V_sb = np.zeros((m*p, rank_sb))
            W_sb = np.zeros((p*n, rank_sb))
            r = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i*m+j, r] = 1.0
                        V_sb[j*p+k, r] = 1.0
                        W_sb[k*n+i, r] = 1.0
                        r += 1
            return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U, "V": V, "W": W}
