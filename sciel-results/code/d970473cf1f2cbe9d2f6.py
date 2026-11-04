"""Program for matrix multiplication 3x4x7 tensor decomposition using Strassen on 2x2 blocks."""
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # We partition the 3x4 matrix into blocks:
    # Rows: 0, 1, 2
    # Cols: 0, 1, 2, 3
    #
    # We can form 2x2 blocks from rows (0,1) and cols (0,1) -> Block A (2x2)
    # We can form 2x2 blocks from rows (0,1) and cols (2,3) -> Block B (2x2)
    # Row 2 is a 1x4 row -> Block C (1x4)
    #
    # The 4x7 matrix is partitioned into:
    # Rows: 0,1,2,3
    # Cols: 0..6
    #
    # For Block A (rows 0,1 of A), we multiply with rows 0,1 of B (4x7).
    # The result is a 2x7 matrix.
    # For Block B (rows 0,1 of A), we multiply with rows 2,3 of B (4x7).
    # The result is a 2x7 matrix.
    # For Block C (row 2 of A), we multiply with all rows 0..3 of B (4x7).
    # The result is a 1x7 matrix.
    #
    # Total output is 2+2+1 = 5 rows, which matches n=3? No, n=3 rows in output.
    # Wait, the output C is 3x7.
    # Row 0 of C comes from row 0 of A (which is in blocks A and B)
    # Row 1 of C comes from row 1 of A (which is in blocks A and B)
    # Row 2 of C comes from row 2 of A (which is block C)
    #
    # So:
    # C[0, :] = A[0, 0:2] @ B[0:2, :] + A[0, 2:4] @ B[2:4, :]
    # C[1, :] = A[1, 0:2] @ B[0:2, :] + A[1, 2:4] @ B[2:4, :]
    # C[2, :] = A[2, 0:2] @ B[0:2, :] + A[2, 2:4] @ B[2:4, :]
    #
    # Let's define:
    # A_top = A[0:2, 0:2] (2x2)
    # A_bot = A[0:2, 2:4] (2x2)
    # A_last = A[2, 0:4] (1x4)
    #
    # B_top = B[0:2, :] (2x7)
    # B_bot = B[2:4, :] (2x7)
    #
    # C[0:2, :] = A_top @ B_top + A_bot @ B_bot  (2x7)
    # C[2, :] = A_last @ B  (1x7)
    #
    # Now, A_top @ B_top is a 2x2 @ 2x7 multiplication.
    # We can use Strassen for the 2x2 part? Strassen is for 2x2 @ 2x2.
    # Here B_top is 2x7, so it's 2x2 @ 2x7.
    # We can think of this as 7 separate 2x2 @ 1x1? No.
    # Actually, 2x2 @ 2x7 can be decomposed into 7 separate 2x2 @ 1x2? No.
    #
    # Let's reconsider. The standard Strassen algorithm reduces a 2x2 @ 2x2 multiplication from 8 multiplications to 7.
    # Here we have 2x2 @ 2x7. This is equivalent to 7 separate 2x2 @ 1x1 multiplications? No, that's not right.
    #
    # Actually, 2x2 @ 2x7 is just 2 rows of A multiplied by 2 columns of B for each of the 7 columns.
    # For each column j of B, we compute A_top @ B_top[:, j] which is a 2x2 @ 1x2? No, B_top[:, j] is 2x1.
    # So it's 2x2 @ 2x1.
    #
    # We can use Strassen for each 2x2 @ 2x1? Strassen is for 2x2 @ 2x2.
    # For 2x2 @ 2x1, we can still use a similar decomposition.
    #
    # Alternatively, we can view the entire problem as a tensor decomposition.
    # The current seed program uses rank = 3*4*7 = 84.
    #
    # Let's try to use Strassen on the 2x2 blocks.
    # A_top is 2x2, B_top is 2x7.
    # We can write B_top as [B_top1, B_top2] where B_top1 is 2x7 and B_top2 is 2x7? No.
    #
    # Let's stick to the 2x2 @ 2x2 case first.
    # If we had A_top (2x2) and B_top2 (2x2), we could use Strassen.
    # But B_top is 2x7.
    #
    # Maybe we can partition the 7 columns into 3+3+1?
    # Then we have three 2x2 @ 2x3 and one 2x2 @ 2x1.
    # This is getting complicated.
    #
    # Let's try a different approach. Use the known rank-7 decomposition for 2x2x2.
    # We can apply this to each 2x2 block of the multiplication.
    #
    # For C[0:2, 0:2] = A_top @ B_top[:, 0:2]
    # This is a 2x2 @ 2x2 multiplication. We can use Strassen (rank 7).
    #
    # For C[0:2, 2:4] = A_top @ B_top[:, 2:4]
    # This is also 2x2 @ 2x2. Rank 7.
    #
    # For C[0:2, 4:6] = A_top @ B_top[:, 4:6]
    # 2x2 @ 2x2. Rank 7.
    #
    # For C[0:2, 6:7] = A_top @ B_top[:, 6:7]
    # 2x2 @ 2x1. This can be done with 4 multiplications (since it's just 2x2 @ 1x1 for each row).
    # Actually, 2x2 @ 2x1 is 4 multiplications.
    #
    # Similarly for A_bot:
    # C[0:2, 0:2] += A_bot @ B_bot[:, 0:2] (rank 7)
    # C[0:2, 2:4] += A_bot @ B_bot[:, 2:4] (rank 7)
    # C[0:2, 4:6] += A_bot @ B_bot[:, 4:6] (rank 7)
    # C[0:2, 6:7] += A_bot @ B_bot[:, 6:7] (rank 4)
    #
    # For A_last (1x4) @ B (4x7):
    # This is 1x4 @ 4x7. It's 7 separate 1x4 @ 4x1 multiplications.
    # Each 1x4 @ 4x1 is 4 multiplications. Total 28.
    #
    # Total rank: 7*3 + 4 + 7*3 + 4 + 28 = 21 + 4 + 21 + 4 + 28 = 78.
    # This matches the previous BET.
    
    # Let's implement this.
    rank = 0
    U = []
    V = []
    W = []
    
    # Strassen 2x2x2 decomposition
    # U patterns (2x2), V patterns (2x2), W patterns (2x2)
    # Based on standard Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)h
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)e
    # M6 = (c-d)(g+h)
    # M7 = (b-a)(e+f)
    #
    # U (input A):
    # M1: [[1,0],[0,1]]
    # M2: [[0,0],[1,1]]
    # M3: [[1,0],[0,0]]
    # M4: [[0,0],[0,1]]
    # M5: [[1,0],[1,0]]
    # M6: [[-1,1],[0,0]]
    # M7: [[0,0],[1,-1]]
    #
    # V (input B):
    # M1: [[1,0],[0,1]]
    # M2: [[0,0],[0,1]]
    # M3: [[0,1],[-1,0]]
    # M4: [[0,0],[1,-1]]
    # M5: [[0,0],[0,1]]
    # M6: [[1,1],[0,0]]
    # M7: [[0,1],[0,1]]
    #
    # W (output C):
    # M1: [[1,0],[0,1]]
    # M2: [[0,0],[1,-1]]
    # M3: [[0,1],[0,1]]
    # M4: [[1,0],[1,0]]
    # M5: [[-1,0],[1,0]]
    # M6: [[0,0],[0,1]]
    # M7: [[1,0],[0,0]]
    
    strassen_U = [
        [[1,0],[0,1]],
        [[0,0],[1,1]],
        [[1,0],[0,0]],
        [[0,0],[0,1]],
        [[1,0],[1,0]],
        [[-1,1],[0,0]],
        [[0,0],[1,-1]]
    ]
    strassen_V = [
        [[1,0],[0,1]],
        [[0,0],[0,1]],
        [[0,1],[-1,0]],
        [[0,0],[1,-1]],
        [[0,0],[0,1]],
        [[1,1],[0,0]],
        [[0,1],[0,1]]
    ]
    strassen_W = [
        [[1,0],[0,1]],
        [[0,0],[1,-1]],
        [[0,1],[0,1]],
        [[1,0],[1,0]],
        [[-1,0],[1,0]],
        [[0,0],[0,1]],
        [[1,0],[0,0]]
    ]
    
    # 2x2 @ 2x1 decomposition (4 multiplications)
    # C[0] = a*e + b*f
    # C[1] = c*e + d*f
    # U patterns: [[1,0],[0,0]], [[0,1],[0,0]], [[0,0],[1,0]], [[0,0],[0,1]]
    # V patterns: [[1],[0]], [[0],[1]], [[1],[0]], [[0],[1]]
    # W patterns: [[1],[0]], [[1],[0]], [[0],[1]], [[0],[1]]
    
    mul2x1_U = [
        [[1,0],[0,0]],
        [[0,1],[0,0]],
        [[0,0],[1,0]],
        [[0,0],[0,1]]
    ]
    mul2x1_V = [
        [[1],[0]],
        [[0],[1]],
        [[1],[0]],
        [[0],[1]]
    ]
    mul2x1_W = [
        [[1],[0]],
        [[1],[0]],
        [[0],[1]],
        [[0],[1]]
    ]
    
    # 1x4 @ 4x1 decomposition (4 multiplications)
    # C = a*e + b*f + c*g + d*h
    # U patterns: [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]
    # V patterns: [1], [1], [1], [1]
    # W patterns: [1], [1], [1], [1]
    
    mul1x4_U = [
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]
    ]
    mul1x4_V = [
        [1], [1], [1], [1]
    ]
    mul1x4_W = [
        [1], [1], [1], [1]
    ]
    
    def add_block(U_local, V_local, W_local, row_offset, col_offset, V_col_offset):
        nonlocal rank, U, V, W
        for r in range(len(U_local)):
            for c in range(len(U_local[0])):
                val = U_local[r][c]
                if val != 0:
                    U.append([0] * (rank + 1))
                    U[-1][rank] = val
                    # Actually, we need to build U, V, W as lists of lists.
                    # Let's restructure.
        
        # This approach is messy. Let's build the matrices directly.
    
    # Let's restart the construction with a cleaner approach.
    # We'll build U, V, W as numpy arrays or lists.
    
    rank = 0
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_strassen(A_row_offset, A_col_offset, B_row_offset, B_col_offset, C_row_offset, C_col_offset):
        nonlocal rank, U_rows, V_rows, W_rows
        for k in range(7):
            # U_local is 2x2, V_local is 2x2, W_local is 2x2
            U_local = strassen_U[k]
            V_local = strassen_V[k]
            W_local = strassen_W[k]
            
            # Add to U
            for i in range(2):
                for j in range(2):
                    val = U_local[i][j]
                    if val != 0:
                        # U row index: (A_row_offset + i) * m + (A_col_offset + j)
                        row_idx = (A_row_offset + i) * m + (A_col_offset + j)
                        if row_idx >= len(U_rows):
                            U_rows.extend([[0] * rank for _ in range(row_idx - len(U_rows) + 1)])
                        while len(U_rows) <= row_idx:
                            U_rows.append([0] * rank)
                        U_rows[row_idx].append(val)
            
            # Add to V
            for i in range(2):
                for j in range(2):
                    val = V_local[i][j]
                    if val != 0:
                        # V row index: (B_row_offset + i) * p + (B_col_offset + j)
                        row_idx = (B_row_offset + i) * p + (B_col_offset + j)
                        if row_idx >= len(V_rows):
                            V_rows.extend([[0] * rank for _ in range(row_idx - len(V_rows) + 1)])
                        while len(V_rows) <= row_idx:
                            V_rows.append([0] * rank)
                        V_rows[row_idx].append(val)
            
            # Add to W
            for i in range(2):
                for j in range(2):
                    val = W_local[i][j]
                    if val != 0:
                        # W row index: (C_col_offset + j) * n + (C_row_offset + i)
                        # Note: W[k*n+i, r] where k is col of C, i is row of C
                        row_idx = (C_col_offset + j) * n + (C_row_offset + i)
                        if row_idx >= len(W_rows):
                            W_rows.extend([[0] * rank for _ in range(row_idx - len(W_rows) + 1)])
                        while len(W_rows) <= row_idx:
                            W_rows.append([0] * rank)
                        W_rows[row_idx].append(val)
            
            rank += 1
    
    def add_mul2x1(A_row_offset, A_col_offset, B_row_offset, B_col_offset, C_row_offset, C_col_offset):
        nonlocal rank, U_rows, V_rows, W_rows
        for k in range(4):
            U_local = mul2x1_U[k]
            V_local = mul2x1_V[k]
            W_local = mul2x1_W[k]
            
            for i in range(2):
                for j in range(2):
                    val = U_local[i][j]
                    if val != 0:
                        row_idx = (A_row_offset + i) * m + (A_col_offset + j)
                        while len(U_rows) <= row_idx:
                            U_rows.append([0] * rank)
                        U_rows[row_idx].append(val)
            
            for i in range(2):
                val = V_local[i][0]
                if val != 0:
                    row_idx = (B_row_offset + i) * p + B_col_offset
                    while len(V_rows) <= row_idx:
                        V_rows.append([0] * rank)
                    V_rows[row_idx].append(val)
            
            for i in range(2):
                val = W_local[i][0]
                if val != 0:
                    row_idx = C_col_offset * n + (C_row_offset + i)
                    while len(W_rows) <= row_idx:
                        W_rows.append([0] * rank)
                    W_rows[row_idx].append(val)
            
            rank += 1
    
    def add_mul1x4(A_row, A_col_offset, B_row_offset, B_col_offset, C_row, C_col_offset):
        nonlocal rank, U_rows, V_rows, W_rows
        for k in range(4):
            U_local = mul1x4_U[k]
            V_local = mul1x4_V[k]
            W_local = mul1x4_W[k]
            
            for j in range(4):
                val = U_local[j]
                if val != 0:
                    row_idx = A_row * m + (A_col_offset + j)
                    while len(U_rows) <= row_idx:
                        U_rows.append([0] * rank)
                    U_rows[row_idx].append(val)
            
            val = V_local[0]
            if val != 0:
                row_idx = (B_row_offset + k) * p + B_col_offset
                while len(V_rows) <= row_idx:
                    V_rows.append([0] * rank)
                V_rows[row_idx].append(val)
            
            val = W_local[0]
            if val != 0:
                row_idx = C_col_offset * n + C_row
                while len(W_rows) <= row_idx:
                    W_rows.append([0] * rank)
                W_rows[row_idx].append(val)
            
            rank += 1
    
    # A_top @ B_top[:, 0:2] -> C[0:2, 0:2]
    add_strassen(0, 0, 0, 0, 0, 0)
    # A_top @ B_top[:, 2:4] -> C[0:2, 2:4]
    add_strassen(0, 0, 0, 2, 0, 2)
    # A_top @ B_top[:, 4:6] -> C[0:2, 4:6]
    add_strassen(0, 0, 0, 4, 0, 4)
    # A_top @ B_top[:, 6:7] -> C[0:2, 6:7]
    add_mul2x1(0, 0, 0, 6, 0, 6)
    
    # A_bot @ B_bot[:, 0:2] -> C[0:2, 0:2]
    add_strassen(0, 2, 2, 0, 0, 0)
    # A_bot @ B_bot[:, 2:4] -> C[0:2, 2:4]
    add_strassen(0, 2, 2, 2, 0, 2)
    # A_bot @ B_bot[:, 4:6] -> C[0:2, 4:6]
    add_strassen(0, 2, 2, 4, 0, 4)
    # A_bot @ B_bot[:, 6:7] -> C[0:2, 6:7]
    add_mul2x1(0, 2, 2, 6, 0, 6)
    
    # A_last @ B[:, 0] -> C[2, 0]
    add_mul1x4(2, 0, 0, 0, 2, 0)
    # A_last @ B[:, 1] -> C[2, 1]
    add_mul1x4(2, 0, 0, 1, 2, 1)
    # A_last @ B[:, 2] -> C[2, 2]
    add_mul1x4(2, 0, 0, 2, 2, 2)
    # A_last @ B[:, 3] -> C[2, 3]
    add_mul1x4(2, 0, 0, 3, 2, 3)
    # A_last @ B[:, 4] -> C[2, 4]
    add_mul1x4(2, 0, 0, 4, 2, 4)
    # A_last @ B[:, 5] -> C[2, 5]
    add_mul1x4(2, 0, 0, 5, 2, 5)
    # A_last @ B[:, 6] -> C[2, 6]
    add_mul1x4(2, 0, 0, 6, 2, 6)
    
    # Pad U, V, W to full size
    U = []
    for i in range(n * m):
        if i < len(U_rows):
            U.append(U_rows[i] + [0] * (rank - len(U_rows[i])))
        else:
            U.append([0] * rank)
    
    V = []
    for i in range(m * p):
        if i < len(V_rows):
            V.append(V_rows[i] + [0] * (rank - len(V_rows[i])))
        else:
            V.append([0] * rank)
    
    W = []
    for i in range(p * n):
        if i < len(W_rows):
            W.append(W_rows[i] + [0] * (rank - len(W_rows[i])))
        else:
            W.append([0] * rank)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
