import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # We decompose 4x4x7 into 4x4x4 and 4x4x3
    # 4x4x4 uses Strassen (Rank 49)
    # 4x4x3 uses 4 blocks of 2x2x3 (Rank 11 each, total 44)
    # Total Rank = 49 + 44 = 93
    
    rank = 93
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # --- Part 1: 4x4x4 (Strassen) ---
    # We need to map the 4x4x4 Strassen decomposition into the global U, V, W
    # Strassen 4x4x4 is recursive: 7 calls to 2x2x2 Strassen.
    # Each 2x2x2 Strassen has rank 7.
    # Total 49 terms.
    
    # Helper to add a 2x2x2 Strassen decomposition at specific offsets
    def add_strassen_222(u_offset, v_offset, w_offset, base_col):
        # u_offset: (i, j) index in the 4x4 A matrix block?
        # Actually, we are building the global U, V, W.
        # The 4x4x4 part corresponds to the first 4 columns of B (k=0..3).
        # So V indices are j*7 + k for k in 0..3.
        # W indices are k*4 + i for k in 0..3.
        
        # Strassen 2x2x2 terms (7 terms)
        # U (2x2), V (2x2), W (2x2)
        # We need to map these to the global indices.
        
        # The 7 Strassen terms for 2x2x2:
        # M1: (a00+a11)(b00+b11)
        # M2: (a10+a11)b00
        # M3: a00(b01-b11)
        # M4: a11(b10-b00)
        # M5: (a00+a01)b11
        # M6: (a10-a00)(b00+b01)
        # M7: (a01-a11)(b10+b11)
        
        # Coefficients for U (A), V (B), W (C)
        # Note: W index in task is k*n + i.
        
        strassen_terms = [
            # U (2x2), V (2x2), W (2x2)
            # 1
            ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]),
            # 2
            ([[0, 0], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [1, 0]]),
            # 3
            ([[1, 0], [0, 0]], [[0, 1], [0, -1]], [[0, 1], [0, 0]]),
            # 4
            ([[0, 0], [0, 1]], [[-1, 0], [1, 0]], [[1, 0], [0, 0]]),
            # 5
            ([[1, 1], [0, 0]], [[0, 0], [0, 1]], [[0, 0], [0, 1]]),
            # 6
            ([[1, 0], [-1, 0]], [[1, 1], [0, 0]], [[1, 0], [0, 0]]),
            # 7
            ([[0, 1], [0, -1]], [[0, 0], [1, 1]], [[0, 1], [0, 0]])
        ]
        
        # This helper is complex to map directly without a general tensor combiner.
        # I will use a simpler approach: generate the full 4x4x4 Strassen decomposition explicitly.
        
    # Given the complexity, I will use the block decomposition for 4x4x4 as well (Rank 56) 
    # to ensure correctness, resulting in Total Rank 56 + 44 = 100.
    # Score 112/100 = 1.12.
    # This is still an improvement.
    
    # Let's refine: Use 8 blocks of 2x2x2 for 4x4x4.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C11 = A11B11 + A12B21
    # C12 = A11B12 + A12B22
    # C21 = A21B11 + A22B21
    # C22 = A21B12 + A22B22
    
    # This requires 8 multiplications of 2x2x2.
    # Each 2x2x2 uses Strassen (Rank 7).
    # Total Rank 56.
    
    # For 4x4x3:
    # Split into 4 blocks of 2x2x3.
    # Each 2x2x3 uses Rank 11 (Strassen 2x2x2 + Schoolbook 2x2x1).
    # Total Rank 44.
    
    # Total Rank 100.
    
    def add_2x2x2_strassen(u_base_i, u_base_j, v_base_j, v_base_k, w_base_k, w_base_i, start_col):
        # u_base_i, u_base_j: offset in A (4x4)
        # v_base_j, v_base_k: offset in B (4x7)
        # w_base_k, w_base_i: offset in C (7x4)
        
        terms = [
            ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]),
            ([[0, 0], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [1, 0]]),
            ([[1, 0], [0, 0]], [[0, 1], [0, -1]], [[0, 1], [0, 0]]),
            ([[0, 0], [0, 1]], [[-1, 0], [1, 0]], [[1, 0], [0, 0]]),
            ([[1, 1], [0, 0]], [[0, 0], [0, 1]], [[0, 0], [0, 1]]),
            ([[1, 0], [-1, 0]], [[1, 1], [0, 0]], [[1, 0], [0, 0]]),
            ([[0, 1], [0, -1]], [[0, 0], [1, 1]], [[0, 1], [0, 0]])
        ]
        
        for r, (u_mat, v_mat, w_mat) in enumerate(terms):
            col = start_col + r
            # U
            for i in range(2):
                for j in range(2):
                    U[u_base_i + i][v_base_j + j][col] = u_mat[i][j]
            # V
            for j in range(2):
                for k in range(2):
                    V[v_base_j + j][v_base_k + k][col] = v_mat[j][k]
            # W
            for k in range(2):
                for i in range(2):
                    W[w_base_k + k][w_base_i + i][col] = w_mat[k][i]
                    
        return start_col + 7

    def add_2x2x3_rank11(u_base_i, u_base_j, v_base_j, v_base_k, w_base_k, w_base_i, start_col):
        # 2x2x3: A(2x2), B(2x3), C(2x3)
        # Split B into B01 (2x2) and B2 (2x1)
        # Part 1: 2x2x2 (Strassen, Rank 7)
        # Part 2: 2x2x1 (Schoolbook, Rank 4)
        
        # Part 1: 2x2x2 for columns 0,1 of B
        next_col = add_2x2x2_strassen(u_base_i, u_base_j, v_base_j, v_base_k, w_base_k, w_base_i, start_col)
        
        # Part 2: 2x2x1 for column 2 of B
        # Schoolbook 2x2x1:
        # c00 = a00*b00 + a01*b10
        # c10 = a10*b00 + a11*b10
        # Terms:
        # 1: U[[1,0],[0,0]], V[[1,0],[0,0]], W[[1,0],[0,0]]
        # 2: U[[0,0],[1,0]], V[[0,1],[0,0]], W[[1,0],[0,0]]
        # 3: U[[0,1],[0,0]], V[[0,0],[1,0]], W[[0,1],[0,0]]
        # 4: U[[0,0],[0,1]], V[[0,0],[0,1]], W[[0,1],[0,0]]
        
        schoolbook_terms = [
            ([[1, 0], [0, 0]], [[1, 0], [0, 0]], [[1, 0], [0, 0]]),
            ([[0, 0], [1, 0]], [[0, 1], [0, 0]], [[1, 0], [0, 0]]),
            ([[0, 1], [0, 0]], [[0, 0], [1, 0]], [[0, 1], [0, 0]]),
            ([[0, 0], [0, 1]], [[0, 0], [0, 1]], [[0, 1], [0, 0]])
        ]
        
        for r, (u_mat, v_mat, w_mat) in enumerate(schoolbook_terms):
            col = next_col + r
            # U
            for i in range(2):
                for j in range(2):
                    U[u_base_i + i][v_base_j + j][col] = u_mat[i][j]
            # V: Note B is 2x3, we are using column 2 (index 2)
            # v_base_k is the offset for k in B. We add 2 to use the 3rd column.
            for j in range(2):
                for k in range(1): # only 1 column
                    V[v_base_j + j][v_base_k + 2 + k][col] = v_mat[j][k]
            # W: Note C is 2x3, we are using column 2 (index 2)
            for k in range(1): # only 1 column
                for i in range(2):
                    W[w_base_k + 2 + k][w_base_i + i][col] = w_mat[k][i]
                    
        return next_col + 4

    # --- Execute Construction ---
    
    # 4x4x4 Part (k=0..3)
    # 4 blocks of 2x2x2
    # A11 (0,0), A12 (0,2), A21 (2,0), A22 (2,2)
    # B11 (0,0), B12 (0,2), B21 (2,0), B22 (2,2)
    
    # C11 = A11 B11 + A12 B21
    col = add_2x2x2_strassen(0, 0, 0, 0, 0, 0, 0) # A11 B11
    col = add_2x2x2_strassen(0, 2, 2, 0, 0, 0, col) # A12 B21
    
    # C12 = A11 B12 + A12 B22
    col = add_2x2x2_strassen(0, 0, 0, 2, 2, 0, col) # A11 B12
    col = add_2x2x2_strassen(0, 2, 2, 2, 2, 0, col) # A12 B22
    
    # C21 = A21 B11 + A22 B21
    col = add_2x2x2_strassen(2, 0, 0, 0, 0, 2, col) # A21 B11
    col = add_2x2x2_strassen(2, 2, 2, 0, 0, 2, col) # A22 B21
    
    # C22 = A21 B12 + A22 B22
    col = add_2x2x2_strassen(2, 0, 0, 2, 2, 2, col) # A21 B12
    col = add_2x2x2_strassen(2, 2, 2, 2, 2, 2, col) # A22 B22
    
    # 4x4x3 Part (k=4..6)
    # 4 blocks of 2x2x3
    # A11 (0,0), A12 (0,2), A21 (2,0), A22 (2,2)
    # B1 (0,4), B2 (2,4)
    
    # C1 = A11 B1 + A12 B2
    col = add_2x2x3_rank11(0, 0, 0, 4, 4, 0, col) # A11 B1
    col = add_2x2x3_rank11(0, 2, 2, 4, 4, 0, col) # A12 B2
    
    # C2 = A21 B1 + A22 B2
    col = add_2x2x3_rank11(2, 0, 0, 4, 4, 2, col) # A21 B1
    col = add_2x2x3_rank11(2, 2, 2, 4, 4, 2, col) # A22 B2
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
