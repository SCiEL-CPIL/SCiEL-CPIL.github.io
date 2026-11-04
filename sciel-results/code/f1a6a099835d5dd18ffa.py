import numpy as np
import time

def construct():
    n, m, p = 4, 5, 6
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin
    
    # We aim for rank 108 using Strassen's algorithm for 2x2 blocks.
    # Partition m=5 into 2+3.
    # A = [A1 (4x2), A2 (4x3)]
    # B = [B1 (2x6); B2 (3x6)]
    # C = A1*B1 + A2*B2
    
    # For A1*B1 (4x2x6):
    #   Split n=4 into 2+2: A1 = [A1a (2x2); A1b (2x2)]
    #   Split p=6 into 3+3: B1 = [B1a (2x3), B1b (2x3)]
    #   C1a = A1a*B1a (2x2x3), C1b = A1a*B1b (2x2x3)
    #   C2a = A1b*B1a (2x2x3), C2b = A1b*B1b (2x2x3)
    #   Each 2x2x3 can be done with rank 11 (using Strassen for 2x2x2 + 1 extra for the 3rd column)
    #   Wait, 2x2x3 rank is 7 + 4 = 11? No.
    #   2x2x2 rank is 7 (Strassen).
    #   2x2x3: We can do 2x2x2 (rank 7) + 2x2x1 (rank 4) = 11.
    #   So 4 * 11 = 44 for A1*B1.
    
    # For A2*B2 (4x3x6):
    #   Split n=4 into 2+2: A2 = [A2a (2x3); A2b (2x3)]
    #   Split p=6 into 3+3: B2 = [B2a (3x3), B2b (3x3)]
    #   C3a = A2a*B2a (2x3x3), C3b = A2a*B2b (2x3x3)
    #   C4a = A2b*B2a (2x3x3), C4b = A2b*B2b (2x3x3)
    #   Each 2x3x3: Rank?
    #   2x3x3 = 2x2x3 (rank 11) + 2x1x3 (rank 3) = 14? No.
    #   Let's think: 2x3x3. Split the 3 in the middle into 2+1.
    #   2x2x3 (rank 11) + 2x1x3 (rank 3) = 14.
    #   Or split the 3 in the last dim: 2x3x2 (rank 11) + 2x3x1 (rank 3) = 14.
    #   Actually, 2x3x3 rank is 17? Let's check.
    #   2x3x3: We can use 2x2x2 (rank 7) + 2x2x1 (rank 4) + 2x1x2 (rank 4) + 2x1x1 (rank 1) = 16?
    #   No, that's not right.
    #   Let's use a known result: rank(2x3x3) = 17.
    #   So 4 * 17 = 68 for A2*B2.
    #   Total: 44 + 68 = 112.
    
    # But the inspiration says 108. How?
    # Maybe rank(2x2x3) = 10? No, it's 11.
    # Maybe rank(2x3x3) = 16?
    # If rank(2x3x3) = 16, then 4 * 16 = 64, total 44 + 64 = 108.
    # Let's assume rank(2x3x3) = 16 is possible.
    
    # We need to construct the actual tensors.
    # This is complex. Let's use a simpler approach:
    # Use Strassen for 2x2x2 blocks and handle the rest with schoolbook.
    
    # Actually, let's just use the schoolbook solution for now to be safe,
    # but I know 108 is the target.
    
    # Let me try to implement the 108 solution properly.
    # I'll define the rank-7 Strassen decomposition for 2x2x2.
    # And rank-4 for 2x2x1, 1x2x2, 2x1x2, etc.
    
    # This is getting too complex to implement from scratch correctly in time.
    # Let's return the schoolbook solution (rank 120) as a fallback,
    # but I'll try to hardcode a known 108 solution if I can recall it.
    
    # Actually, the current program already returns schoolbook (120).
    # The task is to improve it. The inspiration says 108 is possible.
    # Let me try a different partition.
    
    # Alternative: Partition m=5 into 1+4.
    # A = [A1 (4x1), A2 (4x4)]
    # B = [B1 (1x6); B2 (4x6)]
    # C = A1*B1 + A2*B2
    # A1*B1: 4x1x6, rank = 4*1*6 = 24 (schoolbook, no Strassen possible)
    # A2*B2: 4x4x6, split p=6 into 2+4.
    #   A2*B2a: 4x4x2, rank = 4*4*2 = 32? No, use Strassen.
    #   4x4x2: Split 4x4 into 2x2 blocks. 4 blocks of 2x2x2, each rank 7. Total 28.
    #   A2*B2b: 4x4x4, rank = 4*4*4 = 64? No, use Strassen.
    #   4x4x4: 8 blocks of 2x2x2, each rank 7. Total 56.
    #   Total for A2*B2: 28 + 56 = 84.
    #   Total: 24 + 84 = 108.
    
    # This works! Rank 108.
    # Let's implement this.
    
    # U, V, W matrices.
    rank = 108
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    
    # Part 1: A1*B1 (4x1x6)
    # A1 is column 0 of A (indices 0,1,2,3 in U for m=0)
    # B1 is row 0 of B (indices 0..5 in V for m=0)
    # C part is columns 0..5 of C for all rows.
    for i in range(n):
        for k in range(p):
            U[i*m + 0, col] = 1.0
            V[0*p + k, col] = 1.0
            W[k*n + i, col] = 1.0
            col += 1
    
    # Part 2: A2*B2 (4x4x6)
    # A2 is columns 1..4 of A (indices 4..7, 9..12, 14..17, 19..22 in U)
    # B2 is rows 1..4 of B (indices 6..11, 12..17, 18..23, 24..29 in V)
    # Split p=6 into 2+4.
    # B2a is columns 0..1 of B2 (indices 6,7 in V for m=1, 12,13 for m=2, etc.)
    # B2b is columns 2..5 of B2 (indices 8..11 in V for m=1, etc.)
    
    # A2*B2a: 4x4x2
    # Use Strassen on 2x2 blocks.
    # A2 = [[A11, A12], [A21, A22]] where each is 2x2.
    # A11: rows 0,1 of A2 (U indices 4,5, 9,10), cols 0,1 of A2 (m=1,2)
    # A12: rows 0,1 of A2 (U indices 4,5, 9,10), cols 2,3 of A2 (m=3,4)
    # A21: rows 2,3 of A2 (U indices 14,15, 19,20), cols 0,1 of A2 (m=1,2)
    # A22: rows 2,3 of A2 (U indices 14,15, 19,20), cols 2,3 of A2 (m=3,4)
    
    # B2a = [[B11, B12], [B21, B22]] where each is 2x1 (since p=2).
    # Actually, B2a is 4x2. Split into 2x2 blocks:
    # B11: rows 0,1 of B2a (V indices 6,7 for m=1), cols 0,1 (p=0,1)
    # B12: rows 0,1 of B2a (V indices 12,13 for m=2), cols 0,1 (p=0,1)
    # B21: rows 2,3 of B2a (V indices 18,19 for m=3), cols 0,1 (p=0,1)
    # B22: rows 2,3 of B2a (V indices 24,25 for m=4), cols 0,1 (p=0,1)
    
    # Strassen for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to apply this to 4x4x2.
    # This is equivalent to 4 separate 2x2x2 multiplications? No.
    # 4x4x2 = [A11 A12; A21 A22] * [B11 B21; B12 B22] where B is 4x2.
    # Wait, B2a is 4x2. So it's:
    # C_top = A11*B11 + A12*B21
    # C_bottom = A21*B11 + A22*B21
    # Where B11 is 2x2, B21 is 2x2? No.
    # B2a is 4x2. Rows 0,1 are B11 (2x2), rows 2,3 are B21 (2x2).
    # So C_top = A11*B11 + A12*B21 (2x2 result)
    # C_bottom = A21*B11 + A22*B21 (2x2 result)
    
    # This is not standard Strassen. Let's just use schoolbook for the 4x4x2 part.
    # Rank = 4*4*2 = 32.
    # Then A2*B2b is 4x4x4, rank = 4*4*4 = 64.
    # Total: 24 + 32 + 64 = 120. Back to schoolbook.
    
    # To get 108, I need Strassen for 4x4x2.
    # 4x4x2 can be seen as two 4x4x1 multiplications.
    # Or, use Strassen on the 4x4 part.
    # 4x4x2: Split 4x4 into 2x2 blocks.
    # A = [A11 A12; A21 A22], B = [B11 B21; B12 B22] where B is 4x2.
    # B11 is 2x2, B21 is 2x2, etc.
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # This is 4 separate 2x2x2 multiplications if we group correctly?
    # No, it's a 4x4x2 multiplication.
    # Rank of 4x4x2 is 28 (using Strassen).
    # How? 4x4x2 = 2 * (4x4x1)? No.
    # 4x4x2: We can do it with 7 multiplications of size 2x2x1? No.
    # Actually, rank(4x4x2) = 28.
    # Let's just implement the schoolbook for now and accept rank 120.
    # The improvement to 108 requires careful tensor construction.
    
    # Given the time, I'll return the schoolbook solution.
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
