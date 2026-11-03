import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use Strassen for the 2x2 sub-blocks where possible.
    # Matrix A is 3x4. Split rows into [0,1] and [2]. Split cols into [0,1], [2,3].
    # A_top (2x2), A_bot (1x2)
    # A_top_left (2x2), A_top_right (2x2), A_bot_left (1x2), A_bot_right (1x2)
    
    # Matrix B is 4x7. Split rows into [0,1], [2,3]. Split cols into [0,1], [2,3], [4,5], [6].
    # B is composed of blocks B_{ij} where i in {0,1} (row groups of 2), j in {0,1,2,3} (col groups).
    # Actually, let's split B rows into 2 groups of 2.
    # B_top (2x7), B_bot (2x7)
    
    # Result C is 3x7.
    # C_top (2x7) = A_top @ B_top + A_bot @ B_bot ? No, A_bot is 1x2, B_bot is 2x7.
    # C[i, k] = sum_j A[i,j] B[j,k]
    # C[0:2, :] = A[0:2, 0:4] @ B[0:4, :]
    # C[2:3, :] = A[2:3, 0:4] @ B[0:4, :]
    
    # Let's use a more general decomposition.
    # Rank of 3x4x7 is unknown, but likely less than 84.
    # Let's try to use a known low-rank algorithm or just a better partition.
    # Since I don't have a specific low-rank algorithm for 3x4x7, I will stick to a simple improvement:
    # Using Strassen on the 2x2 blocks of the 4x4 part? No, dimensions don't match perfectly.
    
    # Let's try a different approach:
    # The task asks for a "new program... from an approach family that none of the current programs use".
    # Current program uses schoolbook.
    # I will use a "Block Strassen" hybrid.
    
    # Split A (3x4) into A1 (2x2), A2 (2x2), A3 (1x2), A4 (1x2).
    # Split B (4x7) into B1 (2x7), B2 (2x7).
    # C1 (2x7) = A1 @ B1 + A2 @ B2
    # C2 (1x7) = A3 @ B1 + A4 @ B2
    
    # We can compute A1@B1 and A2@B2 using Strassen if B1, B2 were 2x2, but they are 2x7.
    # So we can't directly use Strassen on the full 2x7 blocks.
    # However, we can split B1 and B2 into 2x2 and 2x1 blocks?
    # B1 (2x7) = [B11 (2x2), B12 (2x2), B13 (2x2), B14 (2x1)]
    # B2 (2x7) = [B21 (2x2), B22 (2x2), B23 (2x2), B24 (2x1)]
    
    # C1 (2x7) = A1 @ [B11, B12, B13, B14] + A2 @ [B21, B22, B23, B24]
    # C1[:, 0:2] = A1 @ B11 + A2 @ B21
    # C1[:, 2:4] = A1 @ B12 + A2 @ B22
    # C1[:, 4:6] = A1 @ B13 + A2 @ B23
    # C1[:, 6:7] = A1 @ B14 + A2 @ B24
    
    # For each 2x2 result block (e.g., C1[:, 0:2]), we are computing 2x2 @ 2x2 + 2x2 @ 2x2.
    # This is not a single matrix multiplication, but a sum of two.
    # We can use Strassen for each 2x2 @ 2x2 multiplication.
    # Rank for 2x2@2x2 is 7.
    # So for C1[:, 0:2], we need 7 (for A1@B11) + 7 (for A2@B21) = 14 ranks.
    # For C1[:, 2:4], 14 ranks.
    # For C1[:, 4:6], 14 ranks.
    # For C1[:, 6:7], A1 (2x2) @ B14 (2x1) + A2 (2x2) @ B24 (2x1).
    # 2x2 @ 2x1 is 2x1. Rank of 2x2x1 is 2 (schoolbook) or maybe less?
    # 2x2x1 tensor rank is 2. (c0 = a00*b0 + a01*b1, c1 = a10*b0 + a11*b1).
    # So 2 + 2 = 4 ranks for the last column.
    
    # Total for C1: 14 + 14 + 14 + 4 = 46 ranks.
    
    # Now C2 (1x7) = A3 (1x2) @ B1 (2x7) + A4 (1x2) @ B2 (2x7).
    # A3 @ B1 is 1x7.
    # A3 = [a30, a31]. B1 = [b10, b11, b12, b13, b14, b15, b16] (rows)
    # Actually B1 is 2x7.
    # c2_k = a30 * B1[0, k] + a31 * B1[1, k].
    # This is 1x2 @ 2x7.
    # We can split B1 into 2x2 blocks again.
    # B1 = [B11 (2x2), B12 (2x2), B13 (2x2), B14 (2x1)]
    # c2[0:2] = A3 @ B11 (1x2). Rank 2.
    # c2[2:4] = A3 @ B12 (1x2). Rank 2.
    # c2[4:6] = A3 @ B13 (1x2). Rank 2.
    # c2[6:7] = A3 @ B14 (1x1). Rank 1.
    # Total for A3@B1: 2+2+2+1 = 7 ranks.
    # Similarly for A4@B2: 7 ranks.
    # Total for C2: 14 ranks.
    
    # Total rank: 46 + 14 = 60.
    # This is better than 84.
    
    # Let's construct the matrices.
    # We will create a list of "terms". Each term has U, V, W vectors.
    # Then we sum them up.
    
    U = np.zeros((n*m, 60), dtype=float)
    V = np.zeros((m*p, 60), dtype=float)
    W = np.zeros((p*n, 60), dtype=float)
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 decomposition
    # A (2x2), B (2x2), C (2x2)
    # U_A (4), V_B (4), W_C (4)
    # Strassen 7 terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    def add_strassen(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols, offset):
        nonlocal col
        # A is at A_rows, A_cols in U (index i*m + j)
        # B is at B_rows, B_cols in V (index j*p + k)
        # C is at C_rows, C_cols in W (index k*n + i)
        
        # M1
        U[A_rows[0]*m + A_cols[0], col] += 1
        U[A_rows[1]*m + A_cols[1], col] += 1
        V[B_rows[0]*p + B_cols[0], col] += 1
        V[B_rows[1]*p + B_cols[1], col] += 1
        W[C_cols[0]*n + C_rows[0], col] += 1
        W[C_cols[1]*n + C_rows[1], col] += 1
        W[C_cols[0]*n + C_rows[1], col] += 1
        W[C_cols[1]*n + C_rows[0], col] += 1
        col += 1
        
        # M2
        U[A_rows[1]*m + A_cols[0], col] += 1
        U[A_rows[1]*m + A_cols[1], col] += 1
        V[B_rows[0]*p + B_cols[0], col] += 1
        W[C_cols[0]*n + C_rows[1], col] += 1
        W[C_cols[0]*n + C_rows[0], col] += 1
        W[C_cols[1]*n + C_rows[1], col] += 1
        col += 1
        
        # M3
        U[A_rows[0]*m + A_cols[0], col] += 1
        V[B_rows[1]*p + B_cols[0], col] += 1
        V[B_rows[1]*p + B_cols[1], col] += -1
        W[C_cols[0]*n + C_rows[0], col] += 1
        W[C_cols[1]*n + C_rows[0], col] += 1
        col += 1
        
        # M4
        U[A_rows[1]*m + A_cols[1], col] += 1
        V[B_rows[0]*p + B_cols[1], col] += 1
        V[B_rows[0]*p + B_cols[0], col] += -1
        W[C_cols[1]*n + C_rows[1], col] += 1
        W[C_cols[1]*n + C_rows[0], col] += 1
        col += 1
        
        # M5
        U[A_rows[0]*m + A_cols[0], col] += 1
        U[A_rows[0]*m + A_cols[1], col] += 1
        V[B_rows[1]*p + B_cols[1], col] += 1
        W[C_cols[1]*n + C_rows[0], col] += 1
        W[C_cols[0]*n + C_rows[0], col] += 1
        col += 1
        
        # M6
        U[A_rows[1]*m + A_cols[0], col] += 1
        U[A_rows[0]*m + A_cols[0], col] += -1
        V[B_rows[0]*p + B_cols[0], col] += 1
        V[B_rows[1]*p + B_cols[0], col] += 1
        W[C_cols[0]*n + C_rows[1], col] += 1
        col += 1
        
        # M7
        U[A_rows[0]*m + A_cols[1], col] += 1
        U[A_rows[1]*m + A_cols[1], col] += -1
        V[B_rows[0]*p + B_cols[1], col] += 1
        V[B_rows[1]*p + B_cols[1], col] += 1
        W[C_cols[1]*n + C_rows[0], col] += 1
        col += 1
        
    # Helper to add a 1x2 @ 2x2 decomposition (rank 2)
    # A (1x2), B (2x2), C (1x2)
    # c0 = a0*b00 + a1*b10
    # c1 = a0*b01 + a1*b11
    def add_1x2_2x2(A_row, A_cols, B_rows, B_cols, C_row, C_cols, offset):
        nonlocal col
        # Term 1: a0 * b0
        U[A_row*m + A_cols[0], col] += 1
        V[B_rows[0]*p + B_cols[0], col] += 1
        W[C_cols[0]*n + C_row, col] += 1
        V[B_rows[1]*p + B_cols[0], col] += 0 # wait, b0 is B[0,0], b1 is B[1,0]
        # c0 = a0*b[0,0] + a1*b[1,0]
        # c1 = a0*b[0,1] + a1*b[1,1]
        
        # Let's do it manually to be safe.
        # c0 = a0*b00 + a1*b10
        # c1 = a0*b01 + a1*b11
        
        # Term 1: a0 * b00 -> c0
        U[A_row*m + A_cols[0], col] = 1
        V[B_rows[0]*p + B_cols[0], col] = 1
        W[C_cols[0]*n + C_row, col] = 1
        col += 1
        
        # Term 2: a1 * b10 -> c0
        U[A_row*m + A_cols[1], col] = 1
        V[B_rows[1]*p + B_cols[0], col] = 1
        W[C_cols[0]*n + C_row, col] = 1
        col += 1
        
        # Term 3: a0 * b01 -> c1
        U[A_row*m + A_cols[0], col] = 1
        V[B_rows[0]*p + B_cols[1], col] = 1
        W[C_cols[1]*n + C_row, col] = 1
        col += 1
        
        # Term 4: a1 * b11 -> c1
        U[A_row*m + A_cols[1], col] = 1
        V[B_rows[1]*p + B_cols[1], col] = 1
        W[C_cols[1]*n + C_row, col] = 1
        col += 1
        
    # Helper to add a 1x2 @ 2x1 decomposition (rank 1? No, rank 2)
    # A (1x2), B (2x1), C (1x1)
    # c0 = a0*b0 + a1*b1
    def add_1x2_2x1(A_row, A_cols, B_rows, B_col, C_row, C_col, offset):
        nonlocal col
        # Term 1: a0 * b0
        U[A_row*m + A_cols[0], col] = 1
        V[B_rows[0]*p + B_col, col] = 1
        W[C_col*n + C_row, col] = 1
        col += 1
        
        # Term 2: a1 * b1
        U[A_row*m + A_cols[1], col] = 1
        V[B_rows[1]*p + B_col, col] = 1
        W[C_col*n + C_row, col] = 1
        col += 1
        
    # Indices:
    # A: 3x4
    # A_top: rows 0,1. A_bot: row 2.
    # A_top_left: cols 0,1. A_top_right: cols 2,3.
    # A_bot_left: cols 0,1. A_bot_right: cols 2,3.
    
    # B: 4x7
    # B_top: rows 0,1. B_bot: rows 2,3.
    # B_top_left: cols 0,1. B_top_mid: cols 2,3. B_top_right: cols 4,5. B_top_end: col 6.
    # B_bot_left: cols 0,1. B_bot_mid: cols 2,3. B_bot_right: cols 4,5. B_bot_end: col 6.
    
    # C: 3x7
    # C_top: rows 0,1. C_bot: row 2.
    
    # C_top[:, 0:2] = A_top_left @ B_top_left + A_top_right @ B_bot_left
    # Wait, A is 3x4. A_top is 2x4.
    # A_top = [A_top_left (2x2), A_top_right (2x2)]
    # B_top = [B_top_left (2x2), B_top_mid (2x2), B_top_right (2x2), B_top_end (2x1)]
    # B_bot = [B_bot_left (2x2), B_bot_mid (2x2), B_bot_right (2x2), B_bot_end (2x1)]
    
    # C_top (2x7) = A_top @ B_top + A_bot @ B_bot ? No.
    # C[0:2, :] = A[0:2, :] @ B[:, :]
    # A[0:2, :] is 2x4.
    # B[:, :] is 4x7.
    # A[0:2, :] = [A_top_left (2x2), A_top_right (2x2)]
    # B[:, :] = [B_top (2x7); B_bot (2x7)]
    # C_top = A_top_left @ B_top + A_top_right @ B_bot
    
    # C_bot (1x7) = A_bot @ B
    # A_bot = [A_bot_left (1x2), A_bot_right (1x2)]
    # C_bot = A_bot_left @ B_top + A_bot_right @ B_bot
    
    # So:
    # C_top[:, 0:2] = A_top_left @ B_top[:, 0:2] + A_top_right @ B_bot[:, 0:2]
    # C_top[:, 2:4] = A_top_left @ B_top[:, 2:4] + A_top_right @ B_bot[:, 2:4]
    # C_top[:, 4:6] = A_top_left @ B_top[:, 4:6] + A_top_right @ B_bot[:, 4:6]
    # C_top[:, 6:7] = A_top_left @ B_top[:, 6:7] + A_top_right @ B_bot[:, 6:7]
    
    # C_bot[:, 0:2] = A_bot_left @ B_top[:, 0:2] + A_bot_right @ B_bot[:, 0:2]
    # C_bot[:, 2:4] = A_bot_left @ B_top[:, 2:4] + A_bot_right @ B_bot[:, 2:4]
    # C_bot[:, 4:6] = A_bot_left @ B_top[:, 4:6] + A_bot_right @ B_bot[:, 4:6]
    # C_bot[:, 6:7] = A_bot_left @ B_top[:, 6:7] + A_bot_right @ B_bot[:, 6:7]
    
    # Let's define the blocks.
    # A_top_left: rows [0,1], cols [0,1]
    # A_top_right: rows [0,1], cols [2,3]
    # A_bot_left: rows [2], cols [0,1]
    # A_bot_right: rows [2], cols [2,3]
    
    # B_top_left: rows [0,1], cols [0,1]
    # B_top_mid: rows [0,1], cols [2,3]
    # B_top_right: rows [0,1], cols [4,5]
    # B_top_end: rows [0,1], cols [6]
    
    # B_bot_left: rows [2,3], cols [0,1]
    # B_bot_mid: rows [2,3], cols [2,3]
    # B_bot_right: rows [2,3], cols [4,5]
    # B_bot_end: rows [2,3], cols [6]
    
    # C_top_left: rows [0,1], cols [0,1]
    # C_top_mid: rows [0,1], cols [2,3]
    # C_top_right: rows [0,1], cols [4,5]
    # C_top_end: rows [0,1], cols [6]
    
    # C_bot_left: rows [2], cols [0,1]
    # C_bot_mid: rows [2], cols [2,3]
    # C_bot_right: rows [2], cols [4,5]
    # C_bot_end: rows [2], cols [6]
    
    # 1. C_top_left (2x2) = A_top_left (2x2) @ B_top_left (2x2) + A_top_right (2x2) @ B_bot_left (2x2)
    # Term 1: A_top_left @ B_top_left
    add_strassen([0,1], [0,1], [0,1], [0,1], [0,1], [0,1], 0)
    # Term 2: A_top_right @ B_bot_left
    add_strassen([0,1], [2,3], [2,3], [0,1], [0,1], [0,1], 0)
    
    # 2. C_top_mid (2x2) = A_top_left @ B_top_mid + A_top_right @ B_bot_mid
    add_strassen([0,1], [0,1], [0,1], [2,3], [0,1], [2,3], 0)
    add_strassen([0,1], [2,3], [2,3], [2,3], [0,1], [2,3], 0)
    
    # 3. C_top_right (2x2) = A_top_left @ B_top_right + A_top_right @ B_bot_right
    add_strassen([0,1], [0,1], [0,1], [4,5], [0,1], [4,5], 0)
    add_strassen([0,1], [2,3], [2,3], [4,5], [0,1], [4,5], 0)
    
    # 4. C_top_end (2x1) = A_top_left @ B_top_end + A_top_right @ B_bot_end
    # A_top_left (2x2) @ B_top_end (2x1) -> 2x1
    # Rank 2 for each.
    # c0 = a00*b0 + a01*b1
    # c1 = a10*b0 + a11*b1
    # Term 1: A_top_left @ B_top_end
    # A rows [0,1], cols [0,1]. B rows [0,1], col [6]. C rows [0,1], col [6].
    # c0 = A[0,0]*B[0,6] + A[0,1]*B[1,6]
    # c1 = A[1,0]*B[0,6] + A[1,1]*B[1,6]
    # We can use the 1x2 @ 2x1 helper? No, A is 2x2.
    # Let's just add 4 terms manually.
    # T1: A[0,0] * B[0,6] -> C[0,6]
    U[0*m + 0, col] = 1
    V[0*p + 6, col] = 1
    W[6*n + 0, col] = 1
    col += 1
    # T2: A[0,1] * B[1,6] -> C[0,6]
    U[0*m + 1, col] = 1
    V[1*p + 6, col] = 1
    W[6*n + 0, col] = 1
    col += 1
    # T3: A[1,0] * B[0,6] -> C[1,6]
    U[1*m + 0, col] = 1
    V[0*p + 6, col] = 1
    W[6*n + 1, col] = 1
    col += 1
    # T4: A[1,1] * B[1,6] -> C[1,6]
    U[1*m + 1, col] = 1
    V[1*p + 6, col] = 1
    W[6*n + 1, col] = 1
    col += 1
    
    # Term 2: A_top_right @ B_bot_end
    # A rows [0,1], cols [2,3]. B rows [2,3], col [6]. C rows [0,1], col [6].
    # T5: A[0,2] * B[2,6] -> C[0,6]
    U[0*m + 2, col] = 1
    V[2*p + 6, col] = 1
    W[6*n + 0, col] = 1
    col += 1
    # T6: A[0,3] * B[3,6] -> C[0,6]
    U[0*m + 3, col] = 1
    V[3*p + 6, col] = 1
    W[6*n + 0, col] = 1
    col += 1
    # T7: A[1,2] * B[2,6] -> C[1,6]
    U[1*m + 2, col] = 1
    V[2*p + 6, col] = 1
    W[6*n + 1, col] = 1
    col += 1
    # T8: A[1,3] * B[3,6] -> C[1,6]
    U[1*m + 3, col] = 1
    V[3*p + 6, col] = 1
    W[6*n + 1, col] = 1
    col += 1
    
    # 5. C_bot_left (1x2) = A_bot_left (1x2) @ B_top_left (2x2) + A_bot_right (1x2) @ B_bot_left (2x2)
    # Term 1: A_bot_left @ B_top_left
    # A row [2], cols [0,1]. B rows [0,1], cols [0,1]. C row [2], cols [0,1].
    add_1x2_2x2(2, [0,1], [0,1], [0,1], 2, [0,1], 0)
    # Term 2: A_bot_right @ B_bot_left
    add_1x2_2x2(2, [2,3], [2,3], [0,1], 2, [0,1], 0)
    
    # 6. C_bot_mid (1x2) = A_bot_left @ B_top_mid + A_bot_right @ B_bot_mid
    add_1x2_2x2(2, [0,1], [0,1], [2,3], 2, [2,3], 0)
    add_1x2_2x2(2, [2,3], [2,3], [2,3], 2, [2,3], 0)
    
    # 7. C_bot_right (1x2) = A_bot_left @ B_top_right + A_bot_right @ B_bot_right
    add_1x2_2x2(2, [0,1], [0,1], [4,5], 2, [4,5], 0)
    add_1x2_2x2(2, [2,3], [2,3], [4,5], 2, [4,5], 0)
    
    # 8. C_bot_end (1x1) = A_bot_left @ B_top_end + A_bot_right @ B_bot_end
    # Term 1: A_bot_left @ B_top_end
    # A row [2], cols [0,1]. B rows [0,1], col [6]. C row [2], col [6].
    add_1x2_2x1(2, [0,1], [0,1], 6, 2, 6, 0)
    # Term 2: A_bot_right @ B_bot_end
    add_1x2_2x1(2, [2,3], [2,3], 6, 2, 6, 0)
    
    # Total rank used: col
    # Let's check col.
    # Strassen calls: 6 calls * 7 = 42.
    # C_top_end: 8 terms.
    # 1x2_2x2 calls: 6 calls * 4 = 24.
    # 1x2_2x1 calls: 2 calls * 2 = 4.
    # Total: 42 + 8 + 24 + 4 = 78.
    
    # Wait, my previous estimate was 60. Where is the discrepancy?
    # C_top_left: 14. C_top_mid: 14. C_top_right: 14. C_top_end: 8. Total 50.
    # C_bot_left: 8. C_bot_mid: 8. C_bot_right: 8. C_bot_end: 4. Total 28.
    # Total 78.
    
    # Why did I think 60?
    # C_top: 14+14+14+4 = 46. (I used rank 2 for 2x2@2x1, which is correct, but I used 4 terms for C_top_end, which is rank 4? No, 2x2@2x1 is rank 2. But I added 4 terms. Why?
    # c0 = a00*b0 + a01*b1. c1 = a10*b0 + a11*b1.
    # This is 4 independent terms. Rank is 4?
    # Tensor T[i,j,k] = A[i,j] B[j,k].
    # For 2x2x1, T[0,0,0]=A[0,0]B[0,0], T[0,1,0]=A[0,1]B[1,0], T[1,0,0]=A[1,0]B[0,0], T[1,1,0]=A[1,1]B[1,0].
    # These are 4 independent entries. Rank is 4.
    # So C_top_end is rank 4 (for A_top_left@B_top_end) + 4 (for A_top_right@B_bot_end) = 8.
    # So 50 is correct for C_top.
    
    # C_bot: 1x2 @ 2x2 is rank 4?
    # T[i,j,k] = A[i,j] B[j,k].
    # A is 1x2. B is 2x2.
    # T[0,0,0] = A[0,0]B[0,0]. T[0,0,1] = A[0,0]B[0,1].
    # T[0,1,0] = A[0,1]B[1,0]. T[0,1,1] = A[0,1]B[1,1].
    # 4 independent entries. Rank 4.
    # So C_bot_left is 4+4=8.
    # C_bot_end: 1x2 @ 2x1.
    # T[0,0,0] = A[0,0]B[0,0] + A[0,1]B[1,0].
    # Rank 2.
    # So C_bot_end is 2+2=4.
    # Total C_bot: 8+8+8+4 = 28.
    # Total 78.
    
    # 78 is better than 84.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
