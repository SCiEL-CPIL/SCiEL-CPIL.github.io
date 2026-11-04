"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-88 decomposition for 3x4 * 4x8 matrix multiplication.
    Uses Strassen's algorithm for 2x2x2 blocks.
    """
    n, m, p = DIMENSIONS
    # Rank 88
    # Structure:
    # Split p (8) into p1 (4), p2 (4).
    # For each p-part:
    #   Split n (3) into n1 (2), n2 (1).
    #   For n1 (2x4 * 4x4):
    #     Split m (4) into m1 (2), m2 (2).
    #     A_top = [A11, A12] (2x2 each)
    #     B_part = [B_top; B_bot] (2x4 each)
    #     C_top = A11 * B_top + A12 * B_bot
    #     A11 * B_top is 2x2x4 -> 2 * Strassen(2x2x2) -> 14
    #     A12 * B_bot is 2x2x4 -> 2 * Strassen(2x2x2) -> 14
    #     Total 28
    #   For n2 (1x4 * 4x4):
    #     C_bot = A_bot * B_part
    #     Rank 16
    #   Total per p-part: 44
    # Total: 88
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)] # 3*4 = 12
    V = [[0] * rank for _ in range(m * p)] # 4*8 = 32
    W = [[0] * rank for _ in range(p * n)] # 8*3 = 24
    
    col = 0
    
    def add_strassen_2x2x2(A_rows, A_cols, B_rows, B_cols, W_rows, W_cols):
        """
        Adds a Strassen decomposition for a 2x2x2 block.
        A is at A_rows, A_cols (2x2)
        B is at B_rows, B_cols (2x2)
        W is at W_rows, W_cols (2x2)
        Returns new column index.
        """
        nonlocal col
        # Strassen terms
        # M1 = (A00 + A11) * (B10 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # Indices
        a00, a01 = A_rows[0], A_rows[0] + 1
        a10, a11 = A_rows[1], A_rows[1] + 1
        b00, b01 = B_rows[0], B_rows[0] + 1
        b10, b11 = B_rows[1], B_rows[1] + 1
        w00, w01 = W_rows[0], W_rows[0] + 1
        w10, w11 = W_rows[1], W_rows[1] + 1
        
        # M1
        U[a00][col] += 1
        U[a11][col] += 1
        V[b10][col] += 1
        V[b11][col] += 1
        W[w00][col] += 1
        W[w10][col] += 1
        W[w01][col] -= 1
        W[w11][col] -= 1
        col += 1
        
        # M2
        U[a10][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        W[w00][col] += 1
        W[w10][col] += 1
        W[w01][col] -= 1
        W[w11][col] -= 1
        col += 1
        
        # M3
        U[a00][col] += 1
        V[b01][col] += 1
        V[b11][col] -= 1
        W[w01][col] += 1
        W[w11][col] -= 1
        col += 1
        
        # M4
        U[a11][col] += 1
        V[b10][col] += 1
        V[b00][col] -= 1
        W[w10][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M5
        U[a00][col] += 1
        U[a01][col] += 1
        V[b11][col] += 1
        W[w01][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M6
        U[a10][col] += 1
        U[a00][col] -= 1
        V[b00][col] += 1
        V[b01][col] += 1
        W[w10][col] += 1
        W[w00][col] -= 1
        col += 1
        
        # M7
        U[a01][col] += 1
        U[a11][col] -= 1
        V[b10][col] += 1
        V[b11][col] += 1
        W[w01][col] += 1
        W[w11][col] += 1
        col += 1
        
        return col

    def add_dot_1x4x4(A_row, B_rows, B_cols, W_row, W_cols):
        """
        Adds a rank-16 decomposition for 1x4 * 4x4.
        A is at A_row (1x4)
        B is at B_rows (4x4)
        W is at W_row (1x4)
        """
        nonlocal col
        for j in range(4):
            for k in range(4):
                U[A_row][col] += 1
                V[B_rows[j]][col] += 1
                W[W_row][col] += 1
                col += 1
        return col

    # Process p-part 0 (cols 0-3)
    p_start = 0
    p_end = 4
    
    # n-part 1 (rows 0-1 of A)
    # A_top is A[0:2, 0:4]
    # B_part is B[0:4, 0:4]
    
    # Term 1: A[0:2, 0:2] * B[0:2, 0:4]
    # A_rows: [0, 1] (indices in U)
    # A_cols: [0, 1]
    # B_rows: [0, 1] (indices in V)
    # B_cols: [0, 1, 2, 3] -> Split into two 2x2 blocks
    # W_rows: [0, 1] (indices in W)
    # W_cols: [0, 1, 2, 3]
    
    # Block 1.1: A[0:2, 0:2] * B[0:2, 0:2]
    # U idx: 0, 1 (A[0,0], A[0,1])? No, U is n*m.
    # A[0,0] -> U[0]
    # A[0,1] -> U[1]
    # A[1,0] -> U[2]
    # A[1,1] -> U[3]
    
    # Strassen 1
    add_strassen_2x2x2([0, 2], [0, 1], [0, 2], [0, 1], [0, 1], [0, 1])
    # Strassen 2
    add_strassen_2x2x2([0, 2], [0, 1], [0, 2], [2, 3], [0, 1], [2, 3])
    
    # Term 2: A[0:2, 2:4] * B[2:4, 0:4]
    # A[0,2] -> U[2]
    # A[0,3] -> U[3]
    # A[1,2] -> U[6]
    # A[1,3] -> U[7]
    # B[2,0] -> V[16]
    # B[2,1] -> V[17]
    # B[2,2] -> V[18]
    # B[2,3] -> V[19]
    # B[3,0] -> V[24]
    # ...
    
    # Strassen 3
    add_strassen_2x2x2([2, 6], [2, 3], [16, 24], [0, 1], [0, 1], [0, 1])
    # Strassen 4
    add_strassen_2x2x2([2, 6], [2, 3], [16, 24], [2, 3], [0, 1], [2, 3])
    
    # n-part 2 (row 2 of A)
    # A[2,0:4] * B[0:4, 0:4]
    # A[2,0] -> U[8]
    # B[0,0] -> V[0]
    # W[0,0] -> W[0]
    add_dot_1x4x4(8, [0, 1, 8, 9], [0, 1, 2, 3], 0, [0, 1, 2, 3])
    # Note: V indices for B[0:4, 0:4]
    # B[j,k] -> V[j*8 + k]
    # j=0: 0,1,2,3
    # j=1: 8,9,10,11
    # j=2: 16,17,18,19
    # j=3: 24,25,26,27
    
    # Process p-part 1 (cols 4-7)
    # B[0:4, 4:8]
    # V indices:
    # j=0: 4,5,6,7
    # j=1: 12,13,14,15
    # j=2: 20,21,22,23
    # j=3: 28,29,30,31
    
    # Term 1: A[0:2, 0:2] * B[0:2, 4:8]
    add_strassen_2x2x2([0, 2], [0, 1], [0, 2], [4, 5], [0, 1], [4, 5])
    add_strassen_2x2x2([0, 2], [0, 1], [0, 2], [6, 7], [0, 1], [6, 7])
    
    # Term 2: A[0:2, 2:4] * B[2:4, 4:8]
    add_strassen_2x2x2([2, 6], [2, 3], [16, 24], [4, 5], [0, 1], [4, 5])
    add_strassen_2x2x2([2, 6], [2, 3], [16, 24], [6, 7], [0, 1], [6, 7])
    
    # n-part 2 (row 2 of A)
    add_dot_1x4x4(8, [0, 1, 8, 9], [4, 5, 6, 7], 0, [4, 5, 6, 7])
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
