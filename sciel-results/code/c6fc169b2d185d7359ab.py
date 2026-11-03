import time
import sys

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # We will build the U, V, W matrices for the decomposition.
    # The decomposition is based on splitting A (3x4) and B (4x7) into blocks.
    
    # Blocks:
    # A1 (2x4), A2 (1x4)
    # B1 (4x4), B2 (4x3)
    
    # Sub-blocks for B1: B11 (2x4), B21 (2x4)
    # Sub-blocks for B2: B12 (2x3), B22 (2x3)
    
    # Sub-blocks for B11: B11a (2x2), B11b (2x2)
    # Sub-blocks for B21: B21a (2x2), B21b (2x2)
    # Sub-blocks for B12: B12a (2x2), B12b (2x1)
    # Sub-blocks for B22: B22a (2x2), B22b (2x1)
    
    # Sub-blocks for A1: A11 (2x2), A12 (2x2)
    # Sub-blocks for A2: A21 (1x2), A22 (1x2)
    
    # Multiplications:
    # 1. A11 * B11a (2x2x2) -> Rank 7
    # 2. A11 * B11b (2x2x2) -> Rank 7
    # 3. A12 * B21a (2x2x2) -> Rank 7
    # 4. A12 * B21b (2x2x2) -> Rank 7
    # 5. A11 * B12a (2x2x2) -> Rank 7
    # 6. A11 * B12b (2x2x1) -> Rank 4
    # 7. A12 * B22a (2x2x2) -> Rank 7
    # 8. A12 * B22b (2x2x1) -> Rank 4
    # 9. A21 * B11a (1x2x2) -> Rank 4
    # 10. A21 * B11b (1x2x2) -> Rank 4
    # 11. A22 * B21a (1x2x2) -> Rank 4
    # 12. A22 * B21b (1x2x2) -> Rank 4
    # 13. A21 * B12a (1x2x2) -> Rank 4
    # 14. A21 * B12b (1x2x1) -> Rank 2
    # 15. A22 * B22a (1x2x2) -> Rank 4
    # 16. A22 * B22b (1x2x1) -> Rank 2
    
    # Total Rank: 6*7 + 2*4 + 4*4 + 2*4 + 2*2 = 42 + 8 + 16 + 8 + 4 = 78
    
    rank = 78
    U = [[0] * rank for _ in range(n * m)]  # 12 x 78
    V = [[0] * rank for _ in range(m * p)]  # 28 x 78
    W = [[0] * rank for _ in range(p * n)]  # 21 x 78
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen decomposition
    def add_strassen_222(u_row_offset, v_row_offset, w_row_offset):
        nonlocal col
        # Strassen 7 products for 2x2x2
        # A = [[a11, a12], [a21, a22]]
        # B = [[b11, b12], [b21, b22]]
        # C = [[c11, c12], [c21, c22]]
        
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b12-b22)
        # M4 = a22(b21-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        
        # c11 = M1 + M4 - M5 + M7
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M2 + M3 + M6
        
        # Indices for U (A matrix):
        # a11: u_row_offset + 0
        # a12: u_row_offset + 1
        # a21: u_row_offset + 2
        # a22: u_row_offset + 3
        
        # Indices for V (B matrix):
        # b11: v_row_offset + 0
        # b12: v_row_offset + 1
        # b21: v_row_offset + 2
        # b22: v_row_offset + 3
        
        # Indices for W (C matrix):
        # c11: w_row_offset + 0
        # c12: w_row_offset + 1
        # c21: w_row_offset + 2
        # c22: w_row_offset + 3
        
        # M1
        U[u_row_offset][col] = 1
        U[u_row_offset + 3][col] = 1
        V[v_row_offset][col] = 1
        V[v_row_offset + 3][col] = 1
        W[w_row_offset][col] = 1
        W[w_row_offset + 3][col] = 1
        col += 1
        
        # M2
        U[u_row_offset + 2][col] = 1
        U[u_row_offset + 3][col] = 1
        V[v_row_offset][col] = 1
        W[w_row_offset + 2][col] = 1
        W[w_row_offset + 3][col] = -1
        col += 1
        
        # M3
        U[u_row_offset][col] = 1
        V[v_row_offset + 1][col] = 1
        V[v_row_offset + 3][col] = -1
        W[w_row_offset][col] = -1
        W[w_row_offset + 1][col] = 1
        col += 1
        
        # M4
        U[u_row_offset + 3][col] = 1
        V[v_row_offset + 2][col] = 1
        V[v_row_offset][col] = -1
        W[w_row_offset][col] = 1
        W[w_row_offset + 2][col] = 1
        col += 1
        
        # M5
        U[u_row_offset][col] = 1
        U[u_row_offset + 1][col] = 1
        V[v_row_offset + 3][col] = 1
        W[w_row_offset][col] = -1
        W[w_row_offset + 3][col] = 1
        col += 1
        
        # M6
        U[u_row_offset + 2][col] = 1
        U[u_row_offset][col] = -1
        V[v_row_offset][col] = 1
        V[v_row_offset + 1][col] = 1
        W[w_row_offset + 3][col] = 1
        col += 1
        
        # M7
        U[u_row_offset + 1][col] = 1
        U[u_row_offset + 3][col] = -1
        V[v_row_offset + 2][col] = 1
        V[v_row_offset + 3][col] = 1
        W[w_row_offset][col] = 1
        W[w_row_offset + 3][col] = 1
        col += 1

    # Helper to add a 2x2x1 decomposition (Rank 4)
    def add_221(u_row_offset, v_row_offset, w_row_offset):
        nonlocal col
        # A (2x2), B (2x1), C (2x1)
        # a11, a12; a21, a22
        # b11; b21
        # c11 = a11*b11 + a12*b21
        # c21 = a21*b11 + a22*b21
        
        # Term 1: a11*b11
        U[u_row_offset][col] = 1
        V[v_row_offset][col] = 1
        W[w_row_offset][col] = 1
        col += 1
        
        # Term 2: a12*b21
        U[u_row_offset + 1][col] = 1
        V[v_row_offset + 2][col] = 1
        W[w_row_offset][col] = 1
        col += 1
        
        # Term 3: a21*b11
        U[u_row_offset + 2][col] = 1
        V[v_row_offset][col] = 1
        W[w_row_offset + 1][col] = 1
        col += 1
        
        # Term 4: a22*b21
        U[u_row_offset + 3][col] = 1
        V[v_row_offset + 2][col] = 1
        W[w_row_offset + 1][col] = 1
        col += 1

    # Helper to add a 1x2x2 decomposition (Rank 4)
    def add_122(u_row_offset, v_row_offset, w_row_offset):
        nonlocal col
        # A (1x2), B (2x2), C (1x2)
        # a11, a12
        # b11, b12; b21, b22
        # c11 = a11*b11 + a12*b21
        # c12 = a11*b12 + a12*b22
        
        # Term 1: a11*b11
        U[u_row_offset][col] = 1
        V[v_row_offset][col] = 1
        W[w_row_offset][col] = 1
        col += 1
        
        # Term 2: a12*b21
        U[u_row_offset + 1][col] = 1
        V[v_row_offset + 2][col] = 1
        W[w_row_offset][col] = 1
        col += 1
        
        # Term 3: a11*b12
        U[u_row_offset][col] = 1
        V[v_row_offset + 1][col] = 1
        W[w_row_offset + 1][col] = 1
        col += 1
        
        # Term 4: a12*b22
        U[u_row_offset + 1][col] = 1
        V[v_row_offset + 3][col] = 1
        W[w_row_offset + 1][col] = 1
        col += 1

    # Helper to add a 1x2x1 decomposition (Rank 2)
    def add_121(u_row_offset, v_row_offset, w_row_offset):
        nonlocal col
        # A (1x2), B (2x1), C (1x1)
        # a11, a12
        # b11; b21
        # c11 = a11*b11 + a12*b21
        
        # Term 1: a11*b11
        U[u_row_offset][col] = 1
        V[v_row_offset][col] = 1
        W[w_row_offset][col] = 1
        col += 1
        
        # Term 2: a12*b21
        U[u_row_offset + 1][col] = 1
        V[v_row_offset + 2][col] = 1
        W[w_row_offset][col] = 1
        col += 1

    # Map indices
    # A is 3x4. Rows 0,1 are A1. Row 2 is A2.
    # A11 (2x2): rows 0,1; cols 0,1. U indices: 0,1,2,3
    # A12 (2x2): rows 0,1; cols 2,3. U indices: 4,5,6,7
    # A21 (1x2): row 2; cols 0,1. U indices: 8,9
    # A22 (1x2): row 2; cols 2,3. U indices: 10,11
    
    # B is 4x7.
    # B1 (4x4): cols 0-3.
    # B11 (2x4): rows 0,1; cols 0-3.
    #   B11a (2x2): rows 0,1; cols 0,1. V indices: 0,1,2,3
    #   B11b (2x2): rows 0,1; cols 2,3. V indices: 4,5,6,7
    # B21 (2x4): rows 2,3; cols 0-3.
    #   B21a (2x2): rows 2,3; cols 0,1. V indices: 8,9,10,11
    #   B21b (2x2): rows 2,3; cols 2,3. V indices: 12,13,14,15
    # B2 (4x3): cols 4-6.
    # B12 (2x3): rows 0,1; cols 4-6.
    #   B12a (2x2): rows 0,1; cols 4,5. V indices: 16,17,18,19
    #   B12b (2x1): rows 0,1; col 6. V indices: 20,21
    # B22 (2x3): rows 2,3; cols 4-6.
    #   B22a (2x2): rows 2,3; cols 4,5. V indices: 22,23,24,25
    #   B22b (2x1): rows 2,3; col 6. V indices: 26,27
    
    # C is 3x7.
    # C1 (2x4): rows 0,1; cols 0-3.
    #   C11 (2x2): rows 0,1; cols 0,1. W indices: 0,1,2,3
    #   C12 (2x2): rows 0,1; cols 2,3. W indices: 4,5,6,7
    # C2 (2x3): rows 0,1; cols 4-6.
    #   C21 (2x2): rows 0,1; cols 4,5. W indices: 8,9,10,11
    #   C22 (2x1): rows 0,1; col 6. W indices: 12,13
    # C3 (1x4): row 2; cols 0-3.
    #   C31 (1x2): row 2; cols 0,1. W indices: 14,15
    #   C32 (1x2): row 2; cols 2,3. W indices: 16,17
    # C4 (1x3): row 2; cols 4-6.
    #   C41 (1x2): row 2; cols 4,5. W indices: 18,19
    #   C42 (1x1): row 2; col 6. W indices: 20

    # 1. A11 * B11a -> C11
    add_strassen_222(0, 0, 0)
    # 2. A11 * B11b -> C12
    add_strassen_222(0, 4, 4)
    # 3. A12 * B21a -> C11
    add_strassen_222(4, 8, 0)
    # 4. A12 * B21b -> C12
    add_strassen_222(4, 12, 4)
    
    # 5. A11 * B12a -> C21
    add_strassen_222(0, 16, 8)
    # 6. A11 * B12b -> C22
    add_221(0, 20, 12)
    # 7. A12 * B22a -> C21
    add_strassen_222(4, 22, 8)
    # 8. A12 * B22b -> C22
    add_221(4, 26, 12)
    
    # 9. A21 * B11a -> C31
    add_122(8, 0, 14)
    # 10. A21 * B11b -> C32
    add_122(8, 4, 16)
    # 11. A22 * B21a -> C31
    add_122(10, 8, 14)
    # 12. A22 * B21b -> C32
    add_122(10, 12, 16)
    
    # 13. A21 * B12a -> C41
    add_122(8, 16, 18)
    # 14. A21 * B12b -> C42
    add_121(8, 20, 20)
    # 15. A22 * B22a -> C41
    add_122(10, 22, 18)
    # 16. A22 * B22b -> C42
    add_121(10, 26, 20)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
