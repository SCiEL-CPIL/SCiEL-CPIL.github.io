import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110
    
    nU = n * m  # 12
    nV = m * p  # 32
    nW = p * n  # 24
    
    # Build target tensor indices
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
    
    target_set = set(target_indices)
    
    def full_verify(U, V, W, rank):
        """Check if decomposition is exact - full check"""
        # Check target entries equal 1
        for a, b, c in target_indices:
            val = 0.0
            for cp in range(rank):
                val += U[a][cp] * V[b][cp] * W[c][cp]
            if abs(val - 1.0) > 1e-9:
                return False
        # Check ALL non-target entries equal 0
        for a in range(nU):
            for b in range(nV):
                for c in range(nW):
                    if (a, b, c) in target_set:
                        continue
                    val = 0.0
                    for cp in range(rank):
                        val += U[a][cp] * V[b][cp] * W[c][cp]
                    if abs(val) > 1e-9:
                        return False
        return True
    
    # Build Strassen-based decomposition for 3x4x8
    # Strategy: Use Strassen's 7-product algorithm for 2x2x2 blocks
    # For the 3x4x8 case, we can partition as:
    #   A is 3x4, B is 4x8
    #   We can split A into rows: top 2 rows (2x4) and bottom 1 row (1x4)
    #   We can split B into cols: 4 blocks of 4x2, or use Strassen on 2x2 sub-blocks
    
    # Better approach: Use Strassen's algorithm on 2x2x2 blocks within the multiplication
    # The 3x4x8 multiplication can be viewed as:
    #   C (3x8) = A (3x4) * B (4x8)
    #   Split A into A_top (2x4) and A_bot (1x4)
    #   Split B into B_left (4x4) and B_right (4x4)
    #   C_top = A_top * B (2x8)
    #   C_bot = A_bot * B (1x8)
    #   
    # For C_top = A_top * B:
    #   A_top is 2x4, B is 4x8
    #   Split A_top into 2x2 blocks: A_top = [A00 A01] where A00, A01 are 2x2
    #   Split B into 4x2 blocks: B = [B0 B1 B2 B3] where each is 4x2
    #   Actually, let's use a different decomposition
    #
    # Simpler: Use the known result that 2x2x2 matrix mult has rank 7 (Strassen)
    # and apply it to appropriate sub-blocks
    #
    # For 3x4x8:
    #   We can write C = A*B where A is 3x4, B is 4x8
    #   Partition: A = [A1; A2; A3] (each 1x4), B = [B1 B2 B3 B4] (each 4x2)
    #   Then C[i,j] = sum_k A[i,k] * B[k,j]
    #
    # Let's use a block Strassen approach:
    #   Split A into 2x4 and 1x4 parts
    #   Split B into 4x2, 4x2, 4x2, 4x2 parts (4 blocks of 4x2)
    #   
    # Actually, let me use a cleaner approach:
    #   Use the fact that we can reduce the 2x2x2 tensor from rank 8 to 7
    #   For 3x4x8, we can partition into blocks and apply Strassen where possible
    
    # Direct construction using Strassen for 2x2x2 blocks:
    # The 3x4x8 multiplication tensor can be decomposed by:
    # 1. For the first 2 rows of A (2x4) multiplied by B (4x8):
    #    - This is a 2x4x8 multiplication
    #    - Split B into 4 blocks of 4x2: B = [B1, B2, B3, B4]
    #    - C_top = A_top * B = [A_top*B1, A_top*B2, A_top*B3, A_top*B4]
    #    - Each A_top*Bk is 2x2
    #    - Use Strassen for each 2x2x2: 7 mults each, total 28
    # 2. For the last row of A (1x4) multiplied by B (4x8):
    #    - This is a 1x4x8 multiplication
    #    - C_bot = A_bot * B (1x8)
    #    - This is just 8 scalar multiplications (rank 8)
    # Total: 28 + 8 = 36? No, that's not right.
    #
    # Let me reconsider. The schoolbook rank is 3*4*8 = 96.
    # Using Strassen on 2x2x2 blocks:
    #   Partition A (3x4) into blocks of 2x2 and 1x2
    #   Partition B (4x8) into blocks of 2x2
    #
    # Actually, let's use a simpler but effective approach:
    #   Use the bilinear complexity bound and construct explicitly
    
    # I'll construct a decomposition with rank 88 (as per the knowledge)
    # by using Strassen's algorithm on 2x2x2 sub-problems
    
    # Build the decomposition column by column
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper: Add a column for a single term (i,j,k)
    def add_term(i, j, k):
        a = i * m + j
        b = j * p + k
        c = k * n + i
        U_cols.append({a: 1.0})
        V_cols.append({b: 1.0})
        W_cols.append({c: 1.0})
    
    # Helper: Add a Strassen column
    def add_strassen_column(u_dict, v_dict, w_dict):
        U_cols.append(u_dict)
        V_cols.append(v_dict)
        W_cols.append(w_dict)
    
    # Apply Strassen's 7-product algorithm to 2x2x2 blocks
    # For block starting at row i0, col j0 of A, and row j0, col k0 of B
    # producing C block at row i0, col k0
    
    # Strassen's 7 products for 2x2 matrices:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11B11
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B01 + B10)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def strassen_2x2x2(i0, j0, k0):
        """Add 7 columns for Strassen decomposition of 2x2x2 block"""
        # Indices for A block (rows i0, i0+1; cols j0, j0+1)
        a00 = i0 * m + j0
        a01 = i0 * m + j0 + 1
        a10 = (i0 + 1) * m + j0
        a11 = (i0 + 1) * m + j0 + 1
        
        # Indices for B block (rows j0, j0+1; cols k0, k0+1)
        b00 = j0 * p + k0
        b01 = j0 * p + k0 + 1
        b10 = (j0 + 1) * p + k0
        b11 = (j0 + 1) * p + k0 + 1
        
        # Indices for C block (rows i0, i0+1; cols k0, k0+1)
        c00 = k0 * n + i0
        c01 = k0 * n + i0 + 1
        c10 = (k0 + 1) * n + i0
        c11 = (k0 + 1) * n + i0 + 1
        
        # M1 = (A00 + A11)(B00 + B11)
        u1 = {a00: 1.0, a11: 1.0}
        v1 = {b00: 1.0, b11: 1.0}
        w1 = {c00: 1.0, c11: 1.0}
        add_strassen_column(u1, v1, w1)
        
        # M2 = (A10 + A11)B00
        u2 = {a10: 1.0, a11: 1.0}
        v2 = {b00: 1.0}
        w2 = {c10: 1.0, c11: -1.0}
        add_strassen_column(u2, v2, w2)
        
        # M3 = A00(B10 - B11)
        u3 = {a00: 1.0}
        v3 = {b10: 1.0, b11: -1.0}
        w3 = {c01: 1.0, c11: 1.0}
        add_strassen_column(u3, v3, w3)
        
        # M4 = A11B11
        u4 = {a11: 1.0}
        v4 = {b11: 1.0}
        w4 = {c00: 1.0, c10: 1.0}
        add_strassen_column(u4, v4, w4)
        
        # M5 = (A00 + A01)B11
        u5 = {a00: 1.0, a01: 1.0}
        v5 = {b11: 1.0}
        w5 = {c00: -1.0, c01: 1.0}
        add_strassen_column(u5, v5, w5)
        
        # M6 = (A10 - A00)(B00 + B01)
        u6 = {a10: 1.0, a00: -1.0}
        v6 = {b00: 1.0, b01: 1.0}
        w6 = {c11: 1.0}
        add_strassen_column(u6, v6, w6)
        
        # M7 = (A01 - A11)(B01 + B10)
        u7 = {a01: 1.0, a11: -1.0}
        v7 = {b01: 1.0, b10: 1.0}
        w7 = {c00: 1.0}
        add_strassen_column(u7, v7, w7)
    
    # For 3x4x8:
    # A is 3x4, B is 4x8
    # Split into blocks:
    #   A rows: 2 + 1
    #   B cols: 4 blocks of 2
    #   
    # For the 2x4 part of A times B:
    #   Use Strassen on 2x2x2 blocks
    #   A_top is 2x4, B is 4x8
    #   We can split B into 4 blocks of 4x2
    #   For each 2x4x2 sub-problem, we can further decompose
    #
    # Actually, let's use a more systematic approach:
    #   The 3x4x8 tensor can be decomposed by partitioning into 2x2x2 blocks
    #   where possible, and using scalar mults elsewhere
    
    # Let me build it block by block
    # A (3x4) * B (4x8) = C (3x8)
    
    # Partition:
    #   A = [A_top (2x4); A_bot (1x4)]
    #   B = [B1 (4x2) B2 (4x2) B3 (4x2) B4 (4x2)]
    #   C = [C_top (2x8); C_bot (1x8)]
    #   C_top = A_top * B
    #   C_bot = A_bot * B
    
    # For C_top = A_top (2x4) * B (4x8):
    #   Further split A_top = [A_top_left (2x2) A_top_right (2x2)]
    #   C_top = [A_top_left * B_left (2x4)  A_top_right * B_right (2x4)]
    #   where B_left = B[:, 0:4], B_right = B[:, 4:8]
    #
    # For each 2x2x4 sub-problem, split the output into 2x2 blocks:
    #   2x2x4 = two 2x2x2 blocks
    #
    # So total 2x2x2 blocks for C_top: 4
    # Each uses 7 mults (Strassen)
    # Total for C_top: 4 * 7 = 28
    
    # For C_bot = A_bot (1x4) * B (4x8):
    #   This is 1x4x8, which is 8 scalar mults
    #   Total for C_bot: 8
    
    # Grand total: 28 + 8 = 36? But that seems too low.
    # Let me recount.
    #
    # Actually, I think the issue is that the 2x4x8 decomposition needs more care.
    # Let me just use the known result: rank 88 for 3x4x8 using Strassen-like decomposition.
    #
    # I'll construct it explicitly:
    # - 88 columns total
    # - Use Strassen for 2x2x2 blocks where applicable
    # - Use scalar mults for the rest
    
    # Let me build it properly:
    U_cols.clear()
    V_cols.clear()
    W_cols.clear()
    
    # Use Strassen for the 2x2x2 blocks in the top 2 rows
    # A rows 0,1 times B cols 0-7
    # Split into 4 blocks of 2x2x2:
    #   Block (i0=0, j0=0, k0=0): A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    #   Block (i0=0, j0=0, k0=2): A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    #   Block (i0=0, j0=2, k0=0): A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]  -- wait, this overlaps
    
    # I'm overcomplicating this. Let me just use the schoolbook with one merge.
    # The knowledge says rank 88 is achievable. Let me find a valid decomposition.
    
    # Actually, let me just return schoolbook rank 96 for safety, 
    # since I can't easily verify a rank 88 decomposition in the time available.
    # But the knowledge says 88 was measured to work.
    
    # Let me try to construct rank 88:
    # Use Strassen on 2x2x2 blocks for the first 2 rows, scalar for the last row
    
    U_cols.clear()
    V_cols.clear()
    W_cols.clear()
    
    # For i in {0,1} (top 2 rows of A), j in {0,1,2,3}, k in {0,...,7}:
    # We want to use Strassen where we have 2x2x2 blocks
    
    # Partition the 2x4x8 sub-problem (top 2 rows of A) into 2x2x2 blocks:
    # A_top is 2x4, B is 4x8
    # Split A_top into 2 blocks of 2x2: A_top_left (cols 0-1), A_top_right (cols 2-3)
    # Split B into 4 blocks of 4x2: B_0 (cols 0-1), B_1 (cols 2-3), B_2 (cols 4-5), B_3 (cols 6-7)
    # 
    # C_top = [A_top_left * B_0, A_top_left * B_1, A_top_right * B_2, A_top_right * B_3]
    #         [2x2,               2x2,               2x2,               2x2]
    # 
    # Wait, that's not right. C_top = A_top * B, so:
    # C_top[i,k] = sum_j A_top[i,j] * B[j,k]
    # 
    # If I split into 2x2x2 blocks:
    # For block (i0, j0, k0) where i0 in {0}, j0 in {0,2}, k0 in {0,2,4,6}:
    #   A_block = A[i0:i0+2, j0:j0+2] (2x2)
    #   B_block = B[j0:j0+2, k0:k0+2] (2x2)
    #   C_block = C[i0:i0+2, k0:k0+2] (2x2)
    #   C_block = A_block * B_block
    # 
    # But this only covers part of the multiplication. The full C[i,k] = sum_j A[i,j]*B[j,k]
    # requires summing over all j, not just j0 and j0+1.
    #
    # So the 2x2x2 block decomposition doesn't directly apply to the full 2x4x8 problem
    # because the middle dimension is 4, not 2.
    #
    # I'll just return schoolbook rank 96.
    
    U_cols.clear()
    V_cols.clear()
    W_cols.clear()
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                add_term(i, j, k)
    
    rank = len(U_cols)
    
    # Convert to matrix form
    U = [[0.0] * rank for _ in range(nU)]
    V = [[0.0] * rank for _ in range(nV)]
    W = [[0.0] * rank for _ in range(nW)]
    
    for col_idx in range(rank):
        for a, val in U_cols[col_idx].items():
            U[a][col_idx] = val
        for b, val in V_cols[col_idx].items():
            V[b][col_idx] = val
        for c, val in W_cols[col_idx].items():
            W[c][col_idx] = val
    
    return {"U": U, "V": V, "W": W}
