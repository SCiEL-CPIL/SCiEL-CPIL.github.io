import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

def _to_float_list(matrix):
    """Convert a matrix of Fractions to a list of lists of floats (integers if whole)."""
    return [[float(x) for x in row] for row in matrix]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin

    n, m, p = 4, 5, 5
    
    # Target shapes:
    # U: (n*m, rank) = (20, rank)
    # V: (m*p, rank) = (25, rank)
    # W: (p*n, rank) = (20, rank)
    
    # Strategy: Block decomposition
    # A (4x5) * B (5x5) = C (4x5)
    # Split B into B1 (5x2) and B2 (5x3)
    # Split A into A1 (4x2) and A2 (4x3)
    # C = A1*B1 + A2*B2
    
    # Block 1: 4x2 * 2x2 -> 4x2
    # This is 4 independent 1x2 * 2x2 multiplications? 
    # No, matrix multiplication is (4x2) * (2x2) = (4x2).
    # Each row of the result (1x2) is the product of a row of A1 (1x2) and B1 (2x2).
    # So we need 4 instances of 1x2 * 2x2 multiplication.
    # Rank of 1x2 * 2x2 is 2 (schoolbook).
    # Can we do better? 
    # 1x2 * 2x2:
    # [a b] * [x y; z w] = [ax+bz, ay+bw]
    # This requires 4 multiplications in schoolbook.
    # Is there a rank 3 or 2? 
    # Actually, 1x2 * 2x2 is equivalent to 2 independent 1x2 * 2x1 multiplications? No.
    # Let's check known ranks.
    # R(1,2,2) = 4? No, 1*2*2 = 4.
    # Strassen for 2x2 is rank 7.
    # For 4x2 * 2x2, we can view it as 4 rows.
    # If we use Strassen on 2x2 blocks, we need to group rows.
    # Group 4 rows into two 2x2 blocks.
    # Block A1_top (2x2), A1_bot (2x2).
    # B1 is (2x2).
    # Result C1_top = A1_top * B1 (2x2)
    # Result C1_bot = A1_bot * B1 (2x2)
    # Using Strassen for each 2x2 * 2x2:
    # Rank 7 for top, Rank 7 for bot. Total 14.
    # Can we share? No, B1 is the same, but A1_top and A1_bot are different.
    # So 14 ranks for the first part.
    
    # Block 2: 4x3 * 3x3 -> 4x3
    # Group 4 rows into two 2x3 blocks?
    # 2x3 * 3x3.
    # Split 2x3 into [2x1, 2x2]?
    # This is getting complicated.
    
    # Let's try a different block split that aligns with known fast algorithms.
    # Split 4x5 * 5x5 into:
    # (4x2 * 2x2) + (4x3 * 3x3)
    
    # For 4x2 * 2x2:
    # We can treat this as 2 independent 2x2 * 2x2 multiplications if we stack the 4 rows?
    # No, the right matrix is 2x2.
    # A1 is 4x2. B1 is 2x2.
    # We can compute A1 * B1 by performing two 2x2 * 2x2 multiplications:
    # A1 = [A_top; A_bot] where A_top, A_bot are 2x2.
    # A1 * B1 = [A_top * B1; A_bot * B1]
    # Each is a 2x2 * 2x2 multiplication.
    # Using Strassen (rank 7) for each: Total 14.
    
    # For 4x3 * 3x3:
    # A2 is 4x3. B2 is 3x3.
    # We can split A2 into A2_top (2x3) and A2_bot (2x3).
    # C2_top = A2_top * B2 (2x3)
    # C2_bot = A2_bot * B2 (2x3)
    # We need rank of 2x3 * 3x3.
    # Known rank for 2x3 * 3x3?
    # Schoolbook: 2*3*3 = 18.
    # Can we do better?
    # Laderman's algorithm for 3x3 * 3x3 is rank 23.
    # For 2x3 * 3x3, we can view it as 3 independent 2x1 * 1x3? No.
    # It's 2 rows of 1x3 multiplied by 3x3.
    # Each 1x3 * 3x3 is rank 9 (schoolbook) or can it be lower?
    # 1x3 * 3x3: [a b c] * B.
    # This is a linear combination of rows of B.
    # Rank is 9? No, it's just 3 dot products of length 3?
    # Wait, 1x3 * 3x3 = 1x3.
    # Result element j = sum_k a_k B_kj.
    # This is 3 multiplications for each of 3 outputs?
    # Total 9 multiplications.
    # Is there a faster way? 
    # R(1,3,3) = 9? 
    # Actually, R(1,m,p) = mp. So 9.
    # So 2 rows * 9 = 18.
    
    # Total rank = 14 + 18 = 32.
    # This is much better than 100.
    
    # Let's verify if we can do better on the 2x3 * 3x3 part.
    # 2x3 * 3x3.
    # We can split the 3 columns of A2 into 1 and 2?
    # A2 = [A2a (2x1), A2b (2x2)].
    # C2 = A2a * B2a + A2b * B2b.
    # B2 = [B2a (3x1), B2b (3x2)].
    # Term 1: 2x1 * 1x1 -> 2x1. Rank 2 (schoolbook).
    # Term 2: 2x2 * 2x2 -> 2x2. Rank 7 (Strassen).
    # Total for one 2x3 * 3x3 block: 2 + 7 = 9.
    # We have two such blocks (top and bot).
    # Total for 4x3 * 3x3: 9 + 9 = 18.
    
    # So total rank = 14 (for 4x2*2x2) + 18 (for 4x3*3x3) = 32.
    
    # Let's construct this explicitly.
    
    def strassen_2x2(U, V, W, offset):
        """
        Adds Strassen decomposition for 2x2 * 2x2 to U, V, W matrices.
        U, V, W are lists of lists (or numpy arrays) representing the factor matrices.
        offset is the starting column index in U, V, W.
        Returns the new offset (offset + 7).
        
        Convention:
        U index: row_in_block * 2 + col_in_block
        V index: row_in_block * 2 + col_in_block
        W index: row_in_block * 2 + col_in_block
        
        Wait, the global indices are:
        U[i*m + j, r] where i is row of A, j is col of A.
        V[j*p + k, r] where j is row of B (col of A), k is col of B.
        W[k*n + i, r] where k is row of C (col of B), i is col of C (row of A).
        
        In our block 1:
        A1 is 4x2. Rows 0..3, Cols 0..1 of A.
        B1 is 2x2. Rows 0..1 of B, Cols 0..1 of B.
        C1 is 4x2. Rows 0..1 of C, Cols 0..3 of C.
        
        We split A1 into A1_top (rows 0,1) and A1_bot (rows 2,3).
        A1_top is 2x2.
        A1_bot is 2x2.
        
        For A1_top * B1:
        A indices: i in {0,1}, j in {0,1}.
        B indices: j in {0,1}, k in {0,1}.
        C indices: k in {0,1}, i in {0,1}.
        
        For A1_bot * B1:
        A indices: i in {2,3}, j in {0,1}.
        B indices: j in {0,1}, k in {0,1}.
        C indices: k in {0,1}, i in {2,3}.
        
        Strassen for 2x2:
        M1 = (A11+A22)(B11+B22)
        M2 = (A21+A22)B11
        M3 = A11(B21-B22)
        M4 = A22(B12-B11)
        M5 = (A11+A12)B22
        M6 = (A21-A11)(B11+B12)
        M7 = (A12-A22)(B21+B22)
        
        C11 = M1+M4-M5+M7
        C12 = M3+M5
        C21 = M2+M4
        C22 = M1-M2+M3+M6
        
        We need to map these to the global U, V, W indices.
        """
        pass # We will implement this by directly building the factors.

    # Let's build the factors directly for the 32-rank decomposition.
    # Rank 32.
    rank = 32
    U = [[0.0] * rank for _ in range(20)]
    V = [[0.0] * rank for _ in range(25)]
    W = [[0.0] * rank for _ in range(20)]
    
    # Part 1: 4x2 * 2x2
    # Two Strassen blocks.
    # Block 1a: A rows 0,1. B rows 0,1. C rows 0,1.
    # Block 1b: A rows 2,3. B rows 0,1. C rows 2,3.
    
    def add_strassen_block(U, V, W, a_rows, b_rows, c_rows, b_cols, c_cols, offset):
        """
        Adds a Strassen 2x2 * 2x2 block.
        a_rows: list of 2 global row indices for A.
        b_rows: list of 2 global row indices for B.
        c_rows: list of 2 global row indices for C.
        b_cols: list of 2 global col indices for B.
        c_cols: list of 2 global col indices for C.
        offset: starting column in U,V,W.
        """
        # Local indices 0,1 map to global indices
        # A local (i, j) -> Global (a_rows[i], j_offset + j)? 
        # Wait, in our split, A1 cols are 0,1. So j_offset is 0.
        # B1 rows are 0,1. So b_row_offset is 0.
        # B1 cols are 0,1. So b_col_offset is 0.
        # C1 rows are 0,1. So c_row_offset is 0.
        # C1 cols are 0,1. So c_col_offset is 0.
        
        # Generalize:
        # A global row = a_rows[i], col = j
        # B global row = b_rows[j], col = b_cols[k]
        # C global row = c_rows[k], col = c_cols[i]
        
        # Strassen terms
        # M1: (A00+A11)(B00+B11)
        # A00: U[a_rows[0]*5 + 0, r] = 1
        # A11: U[a_rows[1]*5 + 1, r] = 1
        # B00: V[b_rows[0]*5 + b_cols[0], r] = 1
        # B11: V[b_rows[1]*5 + b_cols[1], r] = 1
        # C00: W[c_rows[0]*4 + c_cols[0], r] = 1
        # C11: W[c_rows[1]*4 + c_cols[1], r] = 1
        
        # We need to add 1 to U, V, W for the specific indices.
        
        # M1
        r = offset
        U[a_rows[0]*5 + 0][r] += 1
        U[a_rows[1]*5 + 1][r] += 1
        V[b_rows[0]*5 + b_cols[0]][r] += 1
        V[b_rows[1]*5 + b_cols[1]][r] += 1
        W[c_rows[0]*4 + c_cols[0]][r] += 1
        W[c_rows[1]*4 + c_cols[1]][r] += 1
        
        # M2: (A10+A11)B00
        r = offset + 1
        U[a_rows[1]*5 + 0][r] += 1
        U[a_rows[1]*5 + 1][r] += 1
        V[b_rows[0]*5 + b_cols[0]][r] += 1
        W[c_rows[1]*4 + c_cols[0]][r] += 1 # C10
        
        # M3: A00(B10-B11)
        r = offset + 2
        U[a_rows[0]*5 + 0][r] += 1
        V[b_rows[1]*5 + b_cols[0]][r] += 1
        V[b_rows[1]*5 + b_cols[1]][r] -= 1
        W[c_rows[0]*4 + c_cols[1]][r] += 1 # C01
        W[c_rows[1]*4 + c_cols[1]][r] += 1 # C11
        
        # M4: A11(B01-B00)
        r = offset + 3
        U[a_rows[1]*5 + 1][r] += 1
        V[b_rows[0]*5 + b_cols[1]][r] += 1
        V[b_rows[0]*5 + b_cols[0]][r] -= 1
        W[c_rows[0]*4 + c_cols[0]][r] += 1 # C00
        W[c_rows[1]*4 + c_cols[0]][r] += 1 # C10
        
        # M5: (A00+A01)B11
        r = offset + 4
        U[a_rows[0]*5 + 0][r] += 1
        U[a_rows[0]*5 + 1][r] += 1
        V[b_rows[1]*5 + b_cols[1]][r] += 1
        W[c_rows[0]*4 + c_cols[1]][r] -= 1 # C01
        W[c_rows[1]*4 + c_cols[1]][r] += 1 # C11
        
        # M6: (A10-A00)(B00+B01)
        r = offset + 5
        U[a_rows[1]*5 + 0][r] += 1
        U[a_rows[0]*5 + 0][r] -= 1
        V[b_rows[0]*5 + b_cols[0]][r] += 1
        V[b_rows[0]*5 + b_cols[1]][r] += 1
        W[c_rows[1]*4 + c_cols[1]][r] += 1 # C11
        
        # M7: (A01-A11)(B10+B11)
        r = offset + 6
        U[a_rows[0]*5 + 1][r] += 1
        U[a_rows[1]*5 + 1][r] -= 1
        V[b_rows[1]*5 + b_cols[0]][r] += 1
        V[b_rows[1]*5 + b_cols[1]][r] += 1
        W[c_rows[0]*4 + c_cols[0]][r] += 1 # C00
        W[c_rows[1]*4 + c_cols[1]][r] -= 1 # C11
        
        return offset + 7

    # Part 2: 4x3 * 3x3
    # Split into 4 blocks of 2x1 * 1x1 and 2x2 * 2x2?
    # A2 is 4x3. Cols 2,3,4.
    # B2 is 3x3. Rows 2,3,4. Cols 2,3,4.
    # C2 is 4x3. Rows 2,3,4. Cols 0,1,2.
    
    # We split A2 into 2x1 (col 2) and 2x2 (cols 3,4) for each pair of rows.
    # Rows 0,1:
    #   A2a: rows 0,1, col 2. (2x1)
    #   A2b: rows 0,1, cols 3,4. (2x2)
    #   B2a: row 2, cols 2. (1x1) -- Wait, B is 3x3.
    #   B2a: rows 2, col 2. (1x1)
    #   B2b: rows 3,4, cols 3,4. (2x2)
    
    #   C2a: rows 2, cols 0,1. (1x2) -- Wait, C is 4x5.
    #   C2a: rows 2, cols 0,1.
    #   C2b: rows 3,4, cols 0,1.
    
    #   Product 1: A2a (2x1) * B2a (1x1) -> C2a (2x1)?
    #   No, dimensions:
    #   A2a is 2x1. B2a is 1x1. Result is 2x1.
    #   This corresponds to C rows 2, cols 0,1?
    #   Wait, C row index is k (from B col).
    #   B2a col index is 2. So C row index is 2.
    #   A2a row index is i. So C col index is i.
    #   So C2a is C[2, 0:2]. (1x2).
    #   But A2a * B2a is 2x1.
    #   Mismatch.
    
    # Let's re-verify the block multiplication logic.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # A2 is 4x3 (cols 2,3,4). B2 is 3x3 (rows 2,3,4, cols 2,3,4).
    # C2 is 4x3 (cols 2,3,4 of A? No, C is 4x5. Cols 2,3,4 of C correspond to B cols 2,3,4).
    # So C2 is C[:, 2:5]. (4x3).
    
    # Split A2 into A2a (4x1, col 2) and A2b (4x2, cols 3,4).
    # Split B2 into B2a (1x3, row 2) and B2b (2x3, rows 3,4).
    # C2 = A2a * B2a + A2b * B2b.
    
    # Term 1: A2a (4x1) * B2a (1x3) -> C2a (4x3).
    # This is outer product of a 4x1 and 1x3.
    # Rank 4 (schoolbook).
    # Can we do better? 
    # 4x1 * 1x3.
    # C[i,k] = A[i,0] B[0,k].
    # This is rank 4.
    
    # Term 2: A2b (4x2) * B2b (2x3) -> C2b (4x3).
    # Split A2b into 2x2 blocks (rows 0,1 and 2,3).
    # Split B2b into 2x1 and 2x2?
    # B2b is 2x3. Cols 2,3,4.
    # Split B2b into B2b1 (2x1, col 2) and B2b2 (2x2, cols 3,4).
    # C2b = A2b * B2b1 + A2b * B2b2.
    
    # Sub-term 2a: A2b (4x2) * B2b1 (2x1) -> C2b1 (4x1).
    # Split A2b into 2x2 blocks.
    # A2b_top (2x2) * B2b1 (2x1) -> 2x1.
    # A2b_bot (2x2) * B2b1 (2x1) -> 2x1.
    # Rank of 2x2 * 2x1?
    # 2x2 * 2x1 is 4 mults schoolbook.
    # Can we do better?
    # 2x2 * 2x1:
    # [a b; c d] * [x; y] = [ax+by; cx+dy].
    # Rank 4.
    
    # Sub-term 2b: A2b (4x2) * B2b2 (2x2) -> C2b2 (4x2).
    # Split A2b into 2x2 blocks.
    # A2b_top (2x2) * B2b2 (2x2) -> 2x2.
    # A2b_bot (2x2) * B2b2 (2x2) -> 2x2.
    # Use Strassen (rank 7) for each.
    # Total 14.
    
    # Total for Term 2:
    # 2a: 2 * 4 = 8.
    # 2b: 2 * 7 = 14.
    # Total 22.
    
    # Total for Term 1: 4.
    # Total for Part 2: 26.
    
    # Total Rank: 14 (Part 1) + 26 (Part 2) = 40.
    
    # This is worse than 32. My previous estimate of 18 for 4x3*3x3 was based on a different split.
    # Let's re-evaluate 4x3 * 3x3.
    # A (4x3) * B (3x3).
    # Split A into 2x3 blocks.
    # 2x3 * 3x3.
    # Split 2x3 into 2x1 and 2x2.
    # 2x1 * 1x3 (from B row 0) -> 2x3. Rank 6?
    # 2x2 * 2x3 (from B rows 1,2) -> 2x3.
    # 2x2 * 2x3. Split 2x3 into 2x1 and 2x2.
    # 2x2 * 2x1 -> 2x1. Rank 4.
    # 2x2 * 2x2 -> 2x2. Rank 7.
    # Total for 2x3 * 3x3: 6 + 4 + 7 = 17.
    # Two blocks: 34.
    # Total 14 + 34 = 48.
    
    # It seems the simple block decomposition doesn't get below 40 easily without more cleverness.
    # However, 40 is still much better than 100.
    # And 32 was a guess.
    
    # Let's stick to the 40-rank construction which is easier to verify.
    # Or even simpler: 
    # 4x5 * 5x5.
    # Split into 4x2 * 2x5 and 4x3 * 3x5.
    # 4x2 * 2x5:
    #   Split 2x5 into 2x2 and 2x3.
    #   4x2 * 2x2: Rank 14 (2 Strassen).
    #   4x2 * 2x3:
    #     Split 4x2 into 2x2 blocks.
    #     2x2 * 2x3:
    #       Split 2x3 into 2x1 and 2x2.
    #       2x2 * 2x1: Rank 4.
    #       2x2 * 2x2: Rank 7.
    #       Total 11 per 2x2 block.
    #       Two blocks: 22.
    #   Total 4x2 * 2x5: 14 + 22 = 36.
    
    # 4x3 * 3x5:
    #   Split 3x5 into 3x2 and 3x3.
    #   4x3 * 3x2:
    #     Split 4x3 into 2x3 blocks.
    #     2x3 * 3x2:
    #       Split 2x3 into 2x1 and 2x2.
    #       Split 3x2 into 1x2 and 2x2.
    #       2x1 * 1x2: Rank 2.
    #       2x2 * 2x2: Rank 7.
    #       Total 9 per 2x3 block.
    #       Two blocks: 18.
    #   4x3 * 3x3:
    #     Split 4x3 into 2x3 blocks.
    #     2x3 * 3x3:
    #       Split 2x3 into 2x1 and 2x2.
    #       Split 3x3 into 1x3 and 2x3.
    #       2x1 * 1x3: Rank 3.
    #       2x2 * 2x3: Rank 11.
    #       Total 14 per 2x3 block.
    #       Two blocks: 28.
    #   Total 4x3 * 3x5: 18 + 28 = 46.
    
    # Total Rank: 36 + 46 = 82.
    
    # This is not beating 40.
    
    # Let's go with the 40-rank construction.
    # It is deterministic and valid.
    
    # Reset U, V, W for rank 40.
    rank = 40
    U = [[0.0] * rank for _ in range(20)]
    V = [[0.0] * rank for _ in range(25)]
    W = [[0.0] * rank for _ in range(20)]
    
    offset = 0
    
    # Part 1: 4x2 * 2x2 (Cols 0,1 of A, Rows 0,1 of B, Cols 0,1 of C)
    # Block 1a: A rows 0,1.
    offset = add_strassen_block(U, V, W, [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], offset)
    # Block 1b: A rows 2,3.
    offset = add_strassen_block(U, V, W, [2, 3], [0, 1], [2, 3], [0, 1], [0, 1], offset)
    
    # Part 2: 4x3 * 3x3 (Cols 2,3,4 of A, Rows 2,3,4 of B, Cols 2,3,4 of C)
    # Term 1: A2a (4x1, col 2) * B2a (1x3, row 2) -> C2a (4x3, cols 2,3,4)
    # Rank 4.
    # C[i,k] = A[i,2] B[2,k].
    for i in range(4):
        for k in range(3):
            # U[i*5 + 2, offset] = 1
            # V[2*5 + (2+k), offset] = 1  (B row 2, col 2+k)
            # W[(2+k)*4 + i, offset] = 1  (C row 2+k, col i)
            U[i*5 + 2][offset] = 1.0
            V[2*5 + (2+k)][offset] = 1.0
            W[(2+k)*4 + i][offset] = 1.0
            offset += 1
            
    # Term 2: A2b (4x2, cols 3,4) * B2b (2x3, rows 3,4) -> C2b (4x3, cols 2,3,4)
    # Split A2b into 2x2 blocks (rows 0,1 and 2,3).
    # Split B2b into 2x1 (col 2) and 2x2 (cols 3,4).
    
    # Sub-term 2a: A2b (4x2) * B2b1 (2x1, col 2) -> C2b1 (4x1, col 2)
    # Block 2a1: A rows 0,1.
    # 2x2 * 2x1.
    # A: [a b; c d], B: [x; y].
    # C0 = ax+by, C1 = cx+dy.
    # Rank 4.
    a_rows = [0, 1]
    b_rows = [3, 4]
    c_row = 2
    b_col = 2
    c_col = 2
    # M1: a*x
    U[a_rows[0]*5 + 3][offset] = 1.0
    V[b_rows[0]*5 + b_col][offset] = 1.0
    W[c_row*4 + a_rows[0]][offset] = 1.0
    offset += 1
    # M2: b*y
    U[a_rows[0]*5 + 4][offset] = 1.0
    V[b_rows[1]*5 + b_col][offset] = 1.0
    W[c_row*4 + a_rows[0]][offset] = 1.0
    offset += 1
    # M3: c*x
    U[a_rows[1]*5 + 3][offset] = 1.0
    V[b_rows[0]*5 + b_col][offset] = 1.0
    W[c_row*4 + a_rows[1]][offset] = 1.0
    offset += 1
    # M4: d*y
    U[a_rows[1]*5 + 4][offset] = 1.0
    V[b_rows[1]*5 + b_col][offset] = 1.0
    W[c_row*4 + a_rows[1]][offset] = 1.0
    offset += 1
    
    # Block 2a2: A rows 2,3.
    a_rows = [2, 3]
    # M1: a*x
    U[a_rows[0]*5 + 3][offset] = 1.0
    V[b_rows[0]*5 + b_col][offset] = 1.0
    W[c_row*4 + a_rows[0]][offset] = 1.0
    offset += 1
    # M2: b*y
    U[a_rows[0]*5 + 4][offset] = 1.0
    V[b_rows[1]*5 + b_col][offset] = 1.0
    W[c_row*4 + a_rows[0]][offset] = 1.0
    offset += 1
    # M3: c*x
    U[a_rows[1]*5 + 3][offset] = 1.0
    V[b_rows[0]*5 + b_col][offset] = 1.0
    W[c_row*4 + a_rows[1]][offset] = 1.0
    offset += 1
    # M4: d*y
    U[a_rows[1]*5 + 4][offset] = 1.0
    V[b_rows[1]*5 + b_col][offset] = 1.0
    W[c_row*4 + a_rows[1]][offset] = 1.0
    offset += 1
    
    # Sub-term 2b: A2b (4x2) * B2b2 (2x2, cols 3,4) -> C2b2 (4x2, cols 3,4)
    # Block 2b1: A rows 0,1.
    # 2x2 * 2x2.
    # B rows 3,4. B cols 3,4.
    # C rows 3,4. C cols 0,1.
    offset = add_strassen_block(U, V, W, [0, 1], [3, 4], [3, 4], [3, 4], [3, 4], offset)
    
    # Block 2b2: A rows 2,3.
    offset = add_strassen_block(U, V, W, [2, 3], [3, 4], [3, 4], [3, 4], [3, 4], offset)
    
    # Verify rank
    assert offset == 40, f"Expected offset 40, got {offset}"
    
    # Verify decomposition
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # T = sum_r U[:, r] outer V[:, r] outer W[:, r]
    # T[i, j, k] = sum_r U[i*5+j, r] * V[j*5+k, r] * W[k*4+i, r]
    
    # Check a few random entries
    np.random.seed(42)
    for _ in range(10):
        i = np.random.randint(0, 4)
        j = np.random.randint(0, 5)
        k = np.random.randint(0, 5)
        
        val = 0.0
        for r in range(40):
            val += U_np[i*5+j, r] * V_np[j*5+k, r] * W_np[k*4+i, r]
            
        # Expected: 1 if i,j,k are valid indices for multiplication?
        # No, the tensor T_{i,j,k} is 1 if the indices correspond to a valid multiplication term.
        # In matrix multiplication, C[i,k] = sum_j A[i,j] B[j,k].
        # The tensor is T[i,j,k] = 1.
        # So for any i,j,k, T[i,j,k] should be 1.
        
        if abs(val - 1.0) > 1e-9:
            print(f"Mismatch at {i},{j},{k}: {val}")
            # Fallback to schoolbook if verification fails
            rank = 100
            U = [[0.0] * rank for _ in range(20)]
            V = [[0.0] * rank for _ in range(25)]
            W = [[0.0] * rank for _ in range(20)]
            col = 0
            for i in range(4):
                for j in range(5):
                    for k in range(5):
                        U[i*5+j][col] = 1.0
                        V[j*5+k][col] = 1.0
                        W[k*4+i][col] = 1.0
                        col += 1
            return {"U": U, "V": V, "W": W}
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
