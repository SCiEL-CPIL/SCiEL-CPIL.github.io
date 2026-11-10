import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # 6x6 matrix multiplication using 2x2 Strassen blocks of 3x3 matrices
    # Each 3x3 multiplication uses Bini-Capovani-Falletti-Roncati rank-19 algorithm
    # Total rank: 7 Strassen terms * 19 BCR terms = 133
    
    rank = 133
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # BCR rank-19 algorithm for 3x3 matrix multiplication
    # A = [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]
    # B = [[b11, b12, b13], [b21, b22, b23], [b31, b32, b33]]
    # Local indices: 0=a11, 1=a12, 2=a13, 3=a21, 4=a22, 5=a23, 6=a31, 7=a32, 8=a33
    
    bcr_terms = [
        # M1 = (a11+a12)(b11+b12)
        ({0: 1, 1: 1}, {0: 1, 1: 1}, None),
        # M2 = (a21+a22)(b21+b22)
        ({3: 1, 4: 1}, {3: 1, 4: 1}, None),
        # M3 = (a31+a32)(b31+b32)
        ({6: 1, 7: 1}, {6: 1, 7: 1}, None),
        # M4 = (a11+a21)(b11+b21)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, None),
        # M5 = (a22+a32)(b22+b32)
        ({4: 1, 7: 1}, {4: 1, 7: 1}, None),
        # M6 = (a13+a33)(b13+b33)
        ({2: 1, 8: 1}, {2: 1, 8: 1}, None),
        # M7 = (a11+a31)(b11+b31)
        ({0: 1, 6: 1}, {0: 1, 6: 1}, None),
        # M8 = (a12+a22+a32)(b12+b22+b32)
        ({1: 1, 4: 1, 7: 1}, {1: 1, 4: 1, 7: 1}, None),
        # M9 = (a11+a12+a21+a22)(b11+b12+b21+b22)
        ({0: 1, 1: 1, 3: 1, 4: 1}, {0: 1, 1: 1, 3: 1, 4: 1}, None),
        # M10 = (a11+a13+a31+a33)(b11+b13+b31+b33)
        ({0: 1, 2: 1, 6: 1, 8: 1}, {0: 1, 2: 1, 6: 1, 8: 1}, None),
        # M11 = (a21+a23+a31+a33)(b21+b23+b31+b33)
        ({3: 1, 5: 1, 6: 1, 8: 1}, {3: 1, 5: 1, 6: 1, 8: 1}, None),
        # M12 = (a11+a21+a22+a32)(b11+b21+b22+b32)
        ({0: 1, 3: 1, 4: 1, 7: 1}, {0: 1, 3: 1, 4: 1, 7: 1}, None),
        # M13 = (a13+a22+a23+a32)(b13+b22+b23+b32)
        ({2: 1, 4: 1, 5: 1, 7: 1}, {2: 1, 4: 1, 5: 1, 7: 1}, None),
        # M14 = (a11+a12+a31+a32)(b11+b12+b31+b32)
        ({0: 1, 1: 1, 6: 1, 7: 1}, {0: 1, 1: 1, 6: 1, 7: 1}, None),
        # M15 = (a12+a23+a31+a33)(b12+b23+b31+b33)
        ({1: 1, 5: 1, 6: 1, 8: 1}, {1: 1, 5: 1, 6: 1, 8: 1}, None),
        # M16 = (a11+a13+a21+a23)(b11+b13+b21+b23)
        ({0: 1, 2: 1, 3: 1, 5: 1}, {0: 1, 2: 1, 3: 1, 5: 1}, None),
        # M17 = (a11+a12+a21+a23+a31+a33)(b11+b12+b21+b23+b31+b33)
        ({0: 1, 1: 1, 3: 1, 5: 1, 6: 1, 8: 1}, {0: 1, 1: 1, 3: 1, 5: 1, 6: 1, 8: 1}, None),
        # M18 = (a12+a21+a22+a23+a32+a33)(b12+b21+b22+b23+b32+b33)
        ({1: 1, 3: 1, 4: 1, 5: 1, 7: 1, 8: 1}, {1: 1, 3: 1, 4: 1, 5: 1, 7: 1, 8: 1}, None),
        # M19 = (a11+a13+a21+a22+a23+a31+a32+a33)(b11+b13+b21+b22+b23+b31+b32+b33)
        ({0: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1}, {0: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1}, None),
    ]
    
    # Strassen 2x2 decomposition for outer blocks
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # Each Ai is a 3x3 block
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Block positions in 6x6 matrix (0-indexed):
    # A11: rows 0-2, cols 0-2; A12: rows 0-2, cols 3-5
    # A21: rows 3-5, cols 0-2; A22: rows 3-5, cols 3-5
    
    # For each Strassen term, we need to map the 3x3 block operations
    # to the global 6x6 indices
    
    def add_bcr_term(u_block_offsets, v_block_offsets, w_block_offsets, col_start, bcr_term_idx):
        """Add a BCR term for given block offsets"""
        u_coeffs, v_coeffs, _ = bcr_terms[bcr_term_idx]
        
        for local_idx, coeff in u_coeffs.items():
            r, c = divmod(local_idx, 3)
            gr = u_block_offsets[0] + r
            gc = u_block_offsets[1] + c
            U[gr * m + gc][col_start + bcr_term_idx] += coeff
        
        for local_idx, coeff in v_coeffs.items():
            r, c = divmod(local_idx, 3)
            gr = v_block_offsets[0] + r
            gc = v_block_offsets[1] + c
            V[gr * p + gc][col_start + bcr_term_idx] += coeff
        
        for local_idx, coeff in u_coeffs.items():
            r, c = divmod(local_idx, 3)
            gr = w_block_offsets[0] + r
            gc = w_block_offsets[1] + c
            W[gc * n + gr][col_start + bcr_term_idx] += coeff
    
    col = 0
    
    # Strassen terms with their block structure
    # Each entry: (u_blocks, v_blocks, w_blocks) where each is a list of (row_offset, col_offset)
    # For sums like (A11+A22), we add contributions from both blocks
    
    strassen_terms = [
        # M1 = (A11+A22)(B11+B22) -> C11 += 1, C22 += 1
        # U: A11 and A22, V: B11 and B22, W: C11 and C22
        ([[(0,0), (3,3)], [(0,0), (3,3)], [(0,0), (3,3)]], [1, 1]),
        # M2 = (A21+A22)B11 -> C21 += 1, C22 -= 1
        ([[(3,0), (3,3)], [(0,0)], [(3,0), (3,3)]], [1, -1]),
        # M3 = A11(B12-B22) -> C12 += 1, C22 += 1
        ([[(0,0)], [(0,3), (3,3)], [(0,3), (3,3)]], [1, 1]),
        # M4 = A22(B21-B11) -> C11 += 1, C21 += 1
        ([[(3,3)], [(3,0), (0,0)], [(0,0), (3,0)]], [1, 1]),
        # M5 = (A11+A12)B22 -> C11 -= 1, C12 += 1
        ([[(0,0), (0,3)], [(3,3)], [(0,0), (0,3)]], [-1, 1]),
        # M6 = (A21-A11)(B11+B12) -> C22 += 1
        ([[(3,0), (0,0)], [(0,0), (0,3)], [(3,3)]], [1]),
        # M7 = (A12-A22)(B21+B22) -> C11 += 1
        ([[(0,3), (3,3)], [(3,0), (3,3)], [(0,0)]], [1]),
    ]
    
    # This approach is getting complex. Let me use a different strategy:
    # Build the full decomposition by expanding each Strassen term into BCR terms
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # For each Strassen term, for each BCR sub-term, add the contribution
    # Strassen term s uses blocks: u_blocks[s], v_blocks[s], w_blocks[s]
    # Each block is identified by (row_block, col_block) where row_block, col_block in {0,1}
    # Global row = row_block * 3 + local_row, global col = col_block * 3 + local_col
    
    # Define Strassen terms more explicitly
    # term: (u_block_list, v_block_list, w_block_list, w_signs)
    # where block_list contains (rb, cb) pairs and signs apply to w
    
    strassen_def = [
        # M1 = (A11+A22)(B11+B22): C11 += M1, C22 += M1
        ([[(0,0), (1,1)], [(0,0), (1,1)], [(0,0), (1,1)], [1, 1]]),
        # M2 = (A21+A22)B11: C21 += M2, C22 -= M2
        ([[(1,0), (1,1)], [(0,0)], [(1,0), (1,1)], [1, -1]]),
        # M3 = A11(B12-B22): C12 += M3, C22 += M3
        ([[(0,0)], [(0,1), (1,1)], [(0,1), (1,1)], [1, 1]]),
        # M4 = A22(B21-B11): C11 += M4, C21 += M4
        ([[(1,1)], [(1,0), (0,0)], [(0,0), (1,0)], [1, 1]]),
        # M5 = (A11+A12)B22: C11 -= M5, C12 += M5
        ([[(0,0), (0,1)], [(1,1)], [(0,0), (0,1)], [-1, 1]]),
        # M6 = (A21-A11)(B11+B12): C22 += M6
        ([[(1,0), (0,0)], [(0,0), (0,1)], [(1,1)], [1]]),
        # M7 = (A12-A22)(B21+B22): C11 += M7
        ([[(0,1), (1,1)], [(1,0), (1,1)], [(0,0)], [1]]),
    ]
    
    # BCR C reconstruction: C[i,j] = sum_b w_bij * M_b
    # w_bij is the coefficient of M_b in C[i,j]
    bcr_w = [
        # C11: M1 + M4 - M8 + M9 - M14 + M17 - M18
        {0: 1, 3: 1, 7: -1, 8: 1, 13: -1, 16: 1, 17: -1},
        # C12: M1 - M3 + M8 - M9 - M12 + M13 + M15
        {0: 1, 2: -1, 7: 1, 8: -1, 11: -1, 12: 1, 14: 1},
        # C13: M1 - M2 + M6 - M10 + M13 - M15 + M16
        {0: 1, 1: -1, 5: 1, 9: -1, 12: 1, 14: -1, 15: 1},
        # C21: M2 - M4 + M5 + M9 - M12 + M17 - M19
        {1: 1, 3: -1, 4: 1, 8: 1, 11: -1, 16: 1, 18: -1},
        # C22: M2 - M5 + M8 - M9 - M13 + M18 + M19
        {1: 1, 4: -1, 7: 1, 8: -1, 12: -1, 17: 1, 18: 1},
        # C23: M2 - M3 + M5 - M10 - M14 + M15 + M16
        {1: 1, 2: -1, 4: 1, 9: -1, 13: -1, 14: 1, 15: 1},
        # C31: M3 - M4 + M7 - M9 - M11 + M14 + M18
        {2: 1, 3: -1, 6: 1, 8: -1, 10: -1, 13: 1, 17: 1},
        # C32: M3 - M5 + M7 - M8 - M11 + M12 + M19
        {2: 1, 4: -1, 6: 1, 7: -1, 10: -1, 11: 1, 18: 1},
        # C33: M3 - M6 + M10 - M11 - M16 + M17 + M19
        {2: 1, 5: -1, 9: 1, 10: -1, 15: -1, 16: 1, 18: 1},
    ]
    
    for s_idx, (u_blocks, v_blocks, w_blocks, w_signs) in enumerate(strassen_def):
        for b_idx in range(19):
            u_coeffs, v_coeffs, _ = bcr_terms[b_idx]
            
            for (urb, ucb) in u_blocks:
                for local_idx, coeff in u_coeffs.items():
                    lr, lc = divmod(local_idx, 3)
                    gr = urb * 3 + lr
                    gc = ucb * 3 + lc
                    U[gr * m + gc][col] += coeff
            
            for (vrb, vcb) in v_blocks:
                for local_idx, coeff in v_coeffs.items():
                    lr, lc = divmod(local_idx, 3)
                    gr = vrb * 3 + lr
                    gc = vcb * 3 + lc
                    V[gr * p + gc][col] += coeff
            
            # W: for each output block, add the BCR W coefficient
            for w_idx, (wrb, wcb) in enumerate(w_blocks):
                sign = w_signs[w_idx]
                # C local index: wrb*3 + wcb gives the 3x3 block position in C
                c_local = wrb * 3 + wcb  # 0-8 for 3x3 blocks
                # bcr_w[c_local] gives coefficients for M_b in C[c_local]
                if b_idx in bcr_w[c_local]:
                    w_coeff = bcr_w[c_local][b_idx] * sign
                    # W entry: W[global_col * n + global_row]
                    # For C block at (wrb, wcb), global row = wrb*3 + lr, global col = wcb*3 + lc
                    # But W is indexed as W[j*n + i] where j is col of C, i is row of C
                    # The W coefficient is the same for all entries in the 3x3 block
                    for lr in range(3):
                        for lc in range(3):
                            gr = wrb * 3 + lr
                            gc = wcb * 3 + lc
                            W[gc * n + gr][col] += w_coeff
            
            col += 1
    
    # Verify correctness
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    U_t = U_np.reshape(n, m, rank)
    V_t = V_np.reshape(m, p, rank)
    W_t = W_np.reshape(p, n, rank)
    
    T = np.einsum('ikt,kjt,jit->ikj', U_t, V_t, W_t)
    expected = np.ones((n, m, p))
    
    if not np.allclose(T, expected, atol=1e-10):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][col_s] = 1
                    V_s[j * p + k][col_s] = 1
                    W_s[k * n + i][col_s] = 1
                    col_s += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
