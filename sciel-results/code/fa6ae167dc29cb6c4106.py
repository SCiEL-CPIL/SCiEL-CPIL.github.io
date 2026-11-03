"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0
    n, m, p = DIMENSIONS  # all 5
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def is_valid(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    # Build rank-98 decomposition using 4+1 block split with Strassen on 4x4x4
    def build_rank98():
        rank = 98
        U = [[0] * rank for _ in range(25)]
        V = [[0] * rank for _ in range(25)]
        W = [[0] * rank for _ in range(25)]
        
        # Strassen 2x2x2: 7 multiplications
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # For 4x4x4, we split into 2x2 blocks and apply Strassen.
        # A = [[A00, A01], [A10, A11]] where each is 2x2
        # B = [[B00, B01], [B10, B11]] where each is 2x2
        # C = [[C00, C01], [C10, C11]] where each is 2x2
        
        # Each Strassen product M_s is a 2x2x2 tensor, which itself has rank 7.
        # So total rank = 7 * 7 = 49.
        
        # Strassen outer level: 7 combinations
        # For each combination, we have inner 2x2x2 Strassen: 7 multiplications
        
        # Outer Strassen definitions:
        # Each outer product s has:
        #   A_comb: linear combo of A blocks (2x2 each)
        #   B_comb: linear combo of B blocks (2x2 each)
        #   C_contrib: how M_s contributes to C blocks
        
        # Let me define the 7 outer Strassen terms:
        # s=0: A_comb = A00+A11, B_comb = B00+B11, C: C00+=M, C11+=M
        # s=1: A_comb = A10+A11, B_comb = B00, C: C10+=M, C11-=M
        # s=2: A_comb = A00, B_comb = B01-B11, C: C01+=M, C11+=M
        # s=3: A_comb = A11, B_comb = B10-B00, C: C00+=M, C10+=M
        # s=4: A_comb = A00+A01, B_comb = B11, C: C00-=M, C01+=M
        # s=5: A_comb = A10-A00, B_comb = B00+B01, C: C11+=M
        # s=6: A_comb = A01-A11, B_comb = B10+B11, C: C00+=M
        
        # For each outer term s, the inner 2x2x2 Strassen gives 7 rank-1 terms.
        # The inner Strassen operates on the 2x2 blocks of A_comb and B_comb.
        
        # Let me define for each outer s:
        #   a_coeffs: dict mapping (block_i, block_j) -> coefficient for A
        #   b_coeffs: dict mapping (block_j, block_k) -> coefficient for B
        #   c_coeffs: dict mapping (block_i, block_k) -> coefficient for C
        
        # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3),
        #           A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
        # In terms of (i,j) indices: A00 = A[0:2,0:2], A01 = A[0:2,2:4], A10 = A[2:4,0:2], A11 = A[2:4,2:4]
        
        # B blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3),
        #           B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3)
        # B[0:2,0:2], B[0:2,2:4], B[2:4,0:2], B[2:4,2:4]
        
        # C blocks: C00 (rows 0-1, cols 0-1), C01 (rows 0-1, cols 2-3),
        #           C10 (rows 2-3, cols 0-1), C11 (rows 2-3, cols 2-3)
        # C[0:2,0:2], C[0:2,2:4], C[2:4,0:2], C[2:4,2:4]
        
        # Inner Strassen for 2x2x2:
        # For a 2x2 A_mat and 2x2 B_mat:
        # A_mat = [[a00, a01], [a10, a11]]
        # B_mat = [[b00, b01], [b10, b11]]
        # C_mat = [[c00, c01], [c10, c11]]
        
        # 7 inner products:
        # m0 = (a00+a11)*(b00+b11)
        # m1 = (a10+a11)*b00
        # m2 = a00*(b01-b11)
        # m3 = a11*(b10-b00)
        # m4 = (a00+a01)*b11
        # m5 = (a10-a00)*(b00+b01)
        # m6 = (a01-a11)*(b10+b11)
        
        # c00 = m0 + m3 - m4 + m6
        # c01 = m2 + m4
        # c10 = m1 + m3
        # c11 = m0 - m1 + m2 + m5
        
        # Inner Strassen rank-1 terms (for the 2x2x2 case):
        # Each m_t is a "scalar" multiplication, so in the tensor:
        # m0: U has a00=1, a11=1; V has b00=1, b11=1; W has c00=1, c11=1
        # etc.
        
        # Let me define the inner Strassen terms as (a_indices, a_coeffs, b_indices, b_coeffs, c_indices, c_coeffs)
        # where indices are (row, col) in the 2x2 block
        
        inner_strassen = [
            # (a_positions, a_coeffs, b_positions, b_coeffs, c_positions, c_coeffs)
            # m0 = (a00+a11)*(b00+b11) -> c00+=1, c11+=1
            ([(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1]),
            # m1 = (a10+a11)*b00 -> c10+=1, c11-=1
            ([(1,0), (1,1)], [1, 1], [(0,0)], [1], [(1,0), (1,1)], [1, -1]),
            # m2 = a00*(b01-b11) -> c01+=1, c11+=1
            ([(0,0)], [1], [(0,1), (1,1)], [1, -1], [(0,1), (1,1)], [1, 1]),
            # m3 = a11*(b10-b00) -> c00+=1, c10+=1
            ([(1,1)], [1], [(1,0), (0,0)], [1, -1], [(0,0), (1,0)], [1, 1]),
            # m4 = (a00+a01)*b11 -> c00-=1, c01+=1
            ([(0,0), (0,1)], [1, 1], [(1,1)], [1], [(0,0), (0,1)], [-1, 1]),
            # m5 = (a10-a00)*(b00+b01) -> c11+=1
            ([(1,0), (0,0)], [1, -1], [(0,0), (0,1)], [1, 1], [(1,1)], [1]),
            # m6 = (a01-a11)*(b10+b11) -> c00+=1
            ([(0,1), (1,1)], [1, -1], [(1,0), (1,1)], [1, 1], [(0,0)], [1]),
        ]
        
        # Outer Strassen terms:
        # s=0: A=A00+A11, B=B00+B11, C: C00+=M, C11+=M
        # s=1: A=A10+A11, B=B00, C: C10+=M, C11-=M
        # s=2: A=A00, B=B01-B11, C: C01+=M, C11+=M
        # s=3: A=A11, B=B10-B00, C: C00+=M, C10+=M
        # s=4: A=A00+A01, B=B11, C: C00-=M, C01+=M
        # s=5: A=A10-A00, B=B00+B01, C: C11+=M
        # s=6: A=A01-A11, B=B10+B11, C: C00+=M
        
        # A block positions: A00=(0,0), A01=(0,1), A10=(1,0), A11=(1,1) in block coords
        # Block (bi, bj) covers rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
        
        outer_strassen = [
            # (a_blocks, a_coeffs, b_blocks, b_coeffs, c_blocks, c_coeffs)
            # s=0
            ([(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1]),
            # s=1
            ([(1,0), (1,1)], [1, 1], [(0,0)], [1], [(1,0), (1,1)], [1, -1]),
            # s=2
            ([(0,0)], [1], [(0,1), (1,1)], [1, -1], [(0,1), (1,1)], [1, 1]),
            # s=3
            ([(1,1)], [1], [(1,0), (0,0)], [1, -1], [(0,0), (1,0)], [1, 1]),
            # s=4
            ([(0,0), (0,1)], [1, 1], [(1,1)], [1], [(0,0), (0,1)], [-1, 1]),
            # s=5
            ([(1,0), (0,0)], [1, -1], [(0,0), (0,1)], [1, 1], [(1,1)], [1]),
            # s=6
            ([(0,1), (1,1)], [1, -1], [(1,0), (1,1)], [1, 1], [(0,0)], [1]),
        ]
        
        col = 0
        for s in range(7):
            a_blocks_s, a_coeffs_s, b_blocks_s, b_coeffs_s, c_blocks_s, c_coeffs_s = outer_strassen[s]
            
            for t in range(7):
                a_pos_t, a_coeff_t, b_pos_t, b_coeff_t, c_pos_t, c_coeff_t = inner_strassen[t]
                
                # For this rank-1 term, fill U, V, W
                # U: for each (a_block, a_coeff_s) and (a_pos_t, a_coeff_t):
                #   A index: row = 2*a_block[0] + a_pos_t[0], col = 2*a_block[1] + a_pos_t[1]
                #   U[row*5 + col][col_idx] += a_coeff_s * a_coeff_t
                
                for (ab_i, ab_j), ac_s in zip(a_blocks_s, a_coeffs_s):
                    for (ap_i, ap_j), ac_t in zip(a_pos_t, a_coeff_t):
                        row = 2 * ab_i + ap_i
                        col_a = 2 * ab_j + ap_j
                        U[row * 5 + col_a][col] += ac_s * ac_t
                
                # V: for each (b_block, b_coeff_s) and (b_pos_t, b_coeff_t):
                #   B index: row = 2*b_block[0] + b_pos_t[0], col = 2*b_block[1] + b_pos_t[1]
                #   V[row*5 + col][col_idx] += b_coeff_s * b_coeff_t
                
                for (bb_i, bb_j), bc_s in zip(b_blocks_s, b_coeffs_s):
                    for (bp_i, bp_j), bc_t in zip(b_pos_t, b_coeff_t):
                        row = 2 * bb_i + bp_i
                        col_b = 2 * bb_j + bp_j
                        V[row * 5 + col_b][col] += bc_s * bc_t
                
                # W: for each (c_block, c_coeff_s) and (c_pos_t, c_coeff_t):
                #   C index: C[k*n+i] where k is col of C, i is row of C
                #   C block (cb_i, cb_j) covers rows 2*cb_i..2*cb_i+1, cols 2*cb_j..2*cb_j+1
                #   C position (cp_i, cp_j): row = 2*cb_i + cp_i, col = 2*cb_j + cp_j
                #   W index: col * 5 + row (since W[k*n+i] with k=col, i=row)
                #   W[row * 5 + col][col_idx] += c_coeff_s * c_coeff_t
                
                for (cb_i, cb_j), cc_s in zip(c_blocks_s, c_coeffs_s):
                    for (cp_i, cp_j), cc_t in zip(c_pos_t, c_coeff_t):
                        row = 2 * cb_i + cp_i
                        col_c = 2 * cb_j + cp_j
                        W[col_c * 5 + row][col] += cc_s * cc_t
                
                col += 1
        
        # Now add the cross terms (49 more terms)
        # 2. A12*B21 (4x1 x 1x4 -> 4x4): outer product, 4 terms
        #    C[i,k] += A[i,4]*B[4,k] for i,k in 0..3
        for i in range(4):
            for k in range(4):
                U[i * 5 + 4][col] = 1  # A[i,4]
                V[4 * 5 + k][col] = 1  # B[4,k]
                W[k * 5 + i][col] = 1  # C[k,i] -> W[k*n+i]
                col += 1
        
        # 3. A11*B12 (4x4 x 4x1 -> 4x1): 16 terms
        #    C[i,4] += A[i,j]*B[j,4] for i,j in 0..3
        for i in range(4):
            for j in range(4):
                U[i * 5 + j][col] = 1  # A[i,j]
                V[j * 5 + 4][col] = 1  # B[j,4]
                W[4 * 5 + i][col] = 1  # C[4,i]... wait
                # C[i,4] means row i, col 4 of C. W[k*n+i] with k=4, i=i -> W[4*5+i]
                col += 1
        
        # 4. A12*B22 (4x1 x 1x1 -> 4x1): 4 terms
        #    C[i,4] += A[i,4]*B[4,4] for i in 0..3
        for i in range(4):
            U[i * 5 + 4][col] = 1  # A[i,4]
            V[4 * 5 + 4][col] = 1  # B[4,4]
            W[4 * 5 + i][col] = 1  # C[i,4] -> W[4*5+i]
            col += 1
        
        # 5. A21*B11 (1x4 x 4x4 -> 1x4): 16 terms
        #    C[4,k] += A[4,j]*B[j,k] for j,k in 0..3
        for j in range(4):
            for k in range(4):
                U[4 * 5 + j][col] = 1  # A[4,j]
                V[j * 5 + k][col] = 1  # B[j,k]
                W[k * 5 + 4][col] = 1  # C[4,k] -> W[k*5+4]
                col += 1
        
        # 6. A22*B21 (1x1 x 1x4 -> 1x4): 4 terms
        #    C[4,k] += A[4,4]*B[4,k] for k in 0..3
        for k in range(4):
            U[4 * 5 + 4][col] = 1  # A[4,4]
            V[4 * 5 + k][col] = 1  # B[4,k]
            W[k * 5 + 4][col] = 1  # C[4,k] -> W[k*5+4]
            col += 1
        
        # 7. A21*B12 (1x4 x 4x1 -> 1x1): 4 terms
        #    C[4,4] += A[4,j]*B[j,4] for j in 0..3
        for j in range(4):
            U[4 * 5 + j][col] = 1  # A[4,j]
            V[j * 5 + 4][col] = 1  # B[j,4]
            W[4 * 5 + 4][col] = 1  # C[4,4] -> W[4*5+4]
            col += 1
        
        # 8. A22*B22 (1x1 x 1x1 -> 1x1): 1 term
        #    C[4,4] += A[4,4]*B[4,4]
        U[4 * 5 + 4][col] = 1
        V[4 * 5 + 4][col] = 1
        W[4 * 5 + 4][col] = 1
        col += 1
        
        assert col == 98, f"Expected 98, got {col}"
        
        return {"U": U, "V": V, "W": W}
    
    if time.time() > DEADLINE:
        return schoolbook()
    
    result = build_rank98()
    
    # Verify
    if is_valid(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
