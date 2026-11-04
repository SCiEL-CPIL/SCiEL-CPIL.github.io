import time
import numpy as np
from fractions import Fraction

def construct():
    """
    Rank-minimal tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Uses Strassen's algorithm (rank 7) for 2x2*2x2 multiplications applied
    to block structures, plus schoolbook for the remaining column.
    
    Strategy:
    - Split B (4x9) into B1 (4x4) + B2 (4x4) + B3 (4x1)
    - For each 4x4 * 4x4: use Strassen with rank 49
    - For 4x4 * 4x1: use schoolbook with rank 16
    - Total rank: 49 + 49 + 16 = 114
    """
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 4, 4, 9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Strassen 2x2*2x2 -> 2x2 terms (7 terms)
    # Each term: (u_dict, v_dict, w_dict) where keys are (row,col) pairs 0-3
    # For 2x2 matrices: indices 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_2x2 = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),      # M1=(a+d)(e+h)
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),            # M2=(c+d)e
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),            # M3=a(f-h)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),            # M4=d(g-e)
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),            # M5=(a+b)h
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),            # M6=(c-a)(e+f)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),            # M7=(b-d)(g+h)
    ]
    
    def add_strassen_4x4(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
        """
        Adds 49 terms for 4x4 * 4x4 using Strassen.
        A: rows A_base_i..A_base_i+3, cols A_base_j..A_base_j+3
        B: rows B_base_j..B_base_j+3, cols B_base_k..B_base_k+3
        C: rows C_base_i..C_base_i+3, cols C_base_k..C_base_k+3
        """
        
        # 4x4*4x4 can be decomposed using 2x2 Strassen applied to blocks
        # A = [[A11, A12], [A21, A22]] where each is 2x2
        # B = [[B11, B12], [B21, B22]] where each is 2x2
        # C = A*B, so C11 = A11*B11 + A12*B21, etc.
        
        # We need 4 block multiplications of 2x2*2x2, each rank 7
        # But we can share some terms!
        
        # Actually, the standard approach: use Strassen recursively
        # 4x4*4x4 = 7 * (2x2*2x2) = 7*7 = 49 terms
        
        # Define the 7 Strassen combinations for the 4x4 level
        # M1 = (A11+A22)*(B11+B21)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B12-B22)
        # M4 = A22*(B21-B11)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Each Mi is a 2x2*2x2 multiplication, so we apply Strassen to each
        
        # For each of the 7 terms, we need to compute a 2x2*2x2 product
        # using the 7-term Strassen algorithm
        
        # Define the A and B block combinations for each term
        # A blocks: (row_block, col_block) where row_block, col_block in {0,1}
        # B blocks: (row_block, col_block) where row_block, col_block in {0,1}
        
        # Term 1: M1 = (A11+A22)*(B11+B21)
        # A factors: A11 (block 0,0) + A22 (block 1,1)
        # B factors: B11 (block 0,0) + B21 (block 1,0)
        # C contribution: C11 (block 0,0) + C22 (block 1,1)
        
        # Let me define this more carefully
        # For each Strassen term at the 4x4 level, we have:
        # - A combination: sum of some 2x2 blocks with coefficients
        # - B combination: sum of some 2x2 blocks with coefficients
        # - C combination: sum of some 2x2 blocks with coefficients
        
        # The 7 terms at 4x4 level:
        # M1: A=(A11+A22), B=(B11+B21), C=(C11+C22)
        # M2: A=(A21+A22), B=B11, C=(C21+C22)  -- wait, let me recheck
        
        # Standard Strassen for 4x4:
        # M1 = (A11+A22)(B11+B21)
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
        
        # Each Mi is a 2x2 matrix, computed as product of two 2x2 matrices
        # So we apply 2x2 Strassen to each Mi
        
        # For each term, define:
        # a_blocks: list of (block_row, block_col, coeff) for A
        # b_blocks: list of (block_row, block_col, coeff) for B
        # c_blocks: list of (block_row, block_col, coeff) for C
        
        terms_4x4 = [
            # M1: A=(A11+A22), B=(B11+B21), contributes to C11 and C22
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,0,1)], [(0,0,1), (1,1,1)]),
            # M2: A=(A21+A22), B=B11, contributes to C21 and C22
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
            # M3: A=A11, B=(B12-B22), contributes to C12 and C22
            ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
            # M4: A=A22, B=(B21-B11), contributes to C11 and C21
            ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
            # M5: A=(A11+A12), B=B22, contributes to C11 and C12
            ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
            # M6: A=(A21-A11), B=(B11+B12), contributes to C22
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
            # M7: A=(A12-A22), B=(B21+B22), contributes to C11
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
        ]
        
        for a_blocks, b_blocks, c_blocks in terms_4x4:
            # For each 2x2 Strassen term within this 4x4 term
            for u_c, v_c, w_c in strassen_2x2:
                u_vec = [0] * 16
                v_vec = [0] * 36
                w_vec = [0] * 36
                
                # Map 2x2 indices to global indices
                # 2x2 index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                # Within a 2x2 block at position (br, bc):
                #   local (0,0) -> global (br*2, bc*2)
                #   local (0,1) -> global (br*2, bc*2+1)
                #   local (1,0) -> global (br*2+1, bc*2)
                #   local (1,1) -> global (br*2+1, bc*2+1)
                
                for loc, coeff in u_c.items():
                    li = loc // 2
                    lj = loc % 2
                    for br, bc, bc_coeff in a_blocks:
                        gi = A_base_i + br * 2 + li
                        gj = A_base_j + bc * 2 + lj
                        u_vec[gi * m + gj] += coeff * bc_coeff
                
                for loc, coeff in v_c.items():
                    lj = loc // 2
                    lk = loc % 2
                    for br, bc, bc_coeff in b_blocks:
                        gj = B_base_j + br * 2 + lj
                        gk = B_base_k + bc * 2 + lk
                        v_vec[gj * p + gk] += coeff * bc_coeff
                
                for loc, coeff in w_c.items():
                    li = loc // 2
                    lk = loc % 2
                    for br, bc, bc_coeff in c_blocks:
                        gi = C_base_i + br * 2 + li
                        gk = C_base_k + bc * 2 + lk
                        w_vec[gk * n + gi] += coeff * bc_coeff
                
                add_term(u_vec, v_vec, w_vec)
    
    # A * B1 (cols 0-3)
    add_strassen_4x4(0, 0, 0, 0, 0, 0)
    
    # A * B2 (cols 4-7)
    add_strassen_4x4(0, 0, 0, 4, 0, 4)
    
    # A * B3 (col 8) - schoolbook
    for i in range(4):
        for j in range(4):
            u_vec = [0] * 16
            v_vec = [0] * 36
            w_vec = [0] * 36
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
    
    rank = len(U_cols)
    U = [[U_cols[col][row] for col in range(rank)] for row in range(16)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(36)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(36)]
    
    if time.time() > deadline:
        pass
    
    return {"U": U, "V": V, "W": W}
