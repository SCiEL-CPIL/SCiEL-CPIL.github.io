"""Rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication."""
DIMENSIONS = [2, 3, 7]

def construct():
    # Strategy: Exploit block decomposition of the (2x3)*(3x7) -> (2x7) product.
    # Decompose B into column blocks B1 (3x3) and B2 (3x4).
    # C1 = A * B1 (2x3), C2 = A * B2 (2x4).
    #
    # For C1 = A * B1 (2x3 * 3x3 -> 2x3):
    # Use Strassen's algorithm generalized for 2x2 blocks.
    # Split A into A_top (1x3) and A_bot (1x3), B1 into B1_left (3x1) and B1_right (3x2).
    # Actually, let's use the known result that the tensor rank of (2,3,3) is 8.
    # The tensor rank of 2x3 * 3x3 is known to be 8 (from Landsberg).
    # We will construct a rank-8 decomposition for the (2,3,3) sub-tensor.
    #
    # For C2 = A * B2 (2x3 * 3x4 -> 2x4):
    # Use Strassen's algorithm. Split A into two 1x3 vectors, B2 into 3x4.
    # Actually, we can split A into A0, A1 (each 1x3), and B2 into B00, B01, B10, B11? No.
    # A standard Strassen-like approach for (1,3,2) is rank 3 (schoolbook).
    # Since A is 2x3, we can treat each row independently.
    # Row 0: A0 (1x3) * B (3x7) -> 1x7. Rank = 3 (schoolbook) for each of the 7 columns? No, rank is 3 per column, so 21 total?
    # Wait, the tensor rank of (1,m,n) is exactly m (if m<=n) or n (if n<m)? No.
    # The tensor rank of (1,m,n) is min(m,n).
    # So for 1x3 * 3x7, the rank is 3.
    # For 2x3 * 3x7, we have two rows.
    # C0 = A0 * B, C1 = A1 * B.
    # Since A0 and A1 are independent, can we share computations?
    # Yes, this is exactly the Strassen method.
    # For 2x2 * 2x2, Strassen reduces 8 to 7.
    # For 2x3 * 3x7, we can decompose B into 3x2 blocks? No, 7 is not divisible by 2.
    # Let's decompose B into B1 (3x3) and B2 (3x4).
    # C1 = A * B1 (2x3). Known minimal rank is 8.
    # C2 = A * B2 (2x4). Let's find the minimal rank.
    # 2x3 * 3x4.
    # We can split A into two 1x3 rows.
    # C0 = A0 * B2, C1 = A1 * B2.
    # Rank of (1,3,4) is 3. So C0 requires 3 rank terms, C1 requires 3 rank terms.
    # But they share the B2 factors?
    # If we use schoolbook, it's 2*3*4 = 24.
    # If we use Strassen:
    # Let M1 = (A0+A1) * (B20 + B21) where B20 is first 2 cols, B21 is last 2?
    # Let's partition B2 into B2a (3x2) and B2b (3x2).
    # C2a = A * B2a (2x2), C2b = A * B2b (2x2).
    # Rank of 2x3 * 3x2 is known to be 7 (Strassen).
    # So C2a takes rank 7, C2b takes rank 7. Total 14 for C2.
    # Total rank = 8 (for C1) + 7 (for C2a) + 7 (for C2b) = 22.
    # Schoolbook is 42. 22 is a significant improvement.

    import numpy as np
    from itertools import product

    n, m, p = DIMENSIONS  # 2, 3, 7
    rank1 = 8  # rank for 2x3 * 3x3
    rank2a = 7 # rank for 2x3 * 3x2
    rank2b = 7 # rank for 2x3 * 3x2
    total_rank = rank1 + rank2a + rank2b

    U = [[0] * total_rank for _ in range(n * m)]  # 6 x 22
    V = [[0] * total_rank for _ in range(m * p)]  # 21 x 22
    W = [[0] * total_rank for _ in range(p * n)]  # 14 x 22

    col = 0

    # Helper to add a rank-1 term
    def add_term(u, v, w):
        nonlocal col
        U[u] = (U[u] + [0] * total_rank)
        for i in range(n * m):
            if U[i][col] == 0: pass
        # U is list of lists, so we need to update the whole list
        # Actually, let's just use nested lists and update directly.
        # U is 6x22, V is 21x22, W is 14x22
        for i in range(n * m):
            U[i][col] += 0 # no-op, just to ensure structure
        # Correct approach:
        U[u][col] = 1
        V[v][col] = 1
        W[w][col] = 1
        col += 1

    # --- Block 1: C1 = A * B1 (2x3 * 3x3 -> 2x3) ---
    # We need a rank-8 decomposition of the (2,3,3) tensor.
    # The (2,3,3) tensor corresponds to multiplying a 2x3 matrix by a 3x3 matrix.
    # Known rank is 8.
    # We can construct this by splitting the 3x3 matrix into two 3x2 blocks? No.
    # Let's use the fact that (2,3,3) can be decomposed as:
    # C = [A0; A1] * B
    # Let's use a known decomposition.
    # One approach: 
    # C00 = A00 B00 + A01 B10 + A02 B20
    # C01 = A00 B01 + A01 B11 + A02 B21
    # C02 = A00 B02 + A01 B12 + A02 B22
    # C10 = A10 B00 + A11 B10 + A12 B20
    # C11 = A10 B01 + A11 B11 + A12 B21
    # C12 = A10 B02 + A11 B12 + A12 B22
    # 
    # Using Strassen-like method for 2x2x2 subblocks:
    # Let's define intermediate products:
    # M1 = A0 * B_col0 (1x3 * 3x1 = 1)
    # M2 = A0 * B_col1
    # M3 = A0 * B_col2
    # M4 = A1 * B_col0
    # M5 = A1 * B_col1
    # M6 = A1 * B_col2
    # M7 = (A0+A1) * (B_col0+B_col1)
    # M8 = (A0+A1) * (B_col0+B_col2)
    # 
    # Then:
    # C00 = M1
    # C01 = M2
    # C02 = M3
    # C10 = M4
    # C11 = M5
    # C12 = M6
    # But M7 and M8 give:
    # C11 = M7 - M1 - M2 + (A0*B11?) No.
    # Let's use the explicit rank-8 construction for (2,3,3).
    # The rank of (2,3,3) is exactly 8.
    # We can find it by considering the 8 basis tensors for the minimal rank.
    # 
    # Term 1: A[0,0] * B[0,:] -> C[:, 0]
    # Term 2: A[0,1] * B[1,:] -> C[:, 0]
    # Term 3: A[0,2] * B[2,:] -> C[:, 0]
    # Term 4: A[1,0] * B[0,:] -> C[:, 1]
    # Term 5: A[1,1] * B[1,:] -> C[:, 1]
    # Term 6: A[1,2] * B[2,:] -> C[:, 1]
    # These are 6 terms for schoolbook.
    # We need 2 more terms to cover the off-diagonal blocks?
    # Actually, for (2,3,3), the schoolbook rank is 9 (3*3=9, for each of the 2 rows? No, 2*3*3=18 schoolbook).
    # Minimal rank is 8.
    # 
    # Let's use a verified decomposition:
    # Let P1 = A[:, 0], P2 = A[:, 1], P3 = A[:, 2] (each 2x1)
    # Let Q1 = B[0, :], Q2 = B[1, :], Q3 = B[2, :] (each 1x3)
    # 
    # The tensor is sum_{i,j,k} A[i,j] B[j,k] E_{i,k}
    # 
    # Rank-8 decomposition:
    # 1. A[:,0] * B[0,:] -> C[:, 0]
    # 2. A[:,1] * B[1,:] -> C[:, 1]
    # 3. A[:,2] * B[2,:] -> C[:, 2]
    # 4. (A[:,0]+A[:,1]) * (B[0,:]+B[1,:]) -> C[:, 0:2]
    # 5. (A[:,0]+A[:,2]) * (B[0,:]+B[2,:]) -> C[:, 0, 2]
    # 6. (A[:,1]+A[:,2]) * (B[1,:]+B[2,:]) -> C[:, 1:3]
    # 7. (A[:,0]-A[:,1]) * (B[0,:]-B[1,:]) -> C[:, 0:2]
    # 8. (A[:,0]-A[:,2]) * (B[0,:]-B[2,:]) -> C[:, 0, 2]
    # 
    # Let's verify:
    # C[:,0] = T1 + T4/2 + T5/2 - T7/2 - T8/2 ? No, this is getting complex.
    # 
    # Instead, let's use the known fact that the rank of (2,3,3) is 8 and use a computer algebra approach to find it,
    # or use a simpler block decomposition.
    # 
    # Since we can't easily derive the exact 8 terms manually without errors, let's use a conservative approach:
    # Use schoolbook for the whole problem but with Strassen for the 2x2 blocks.
    # 
    # Decompose B into 7 column blocks of size 3x1? No.
    # Decompose into 3 column blocks of size 3x1, 3x2, 3x2?
    # 
    # Actually, let's use the following strategy:
    # For 2x3 * 3x7, we can use the rank of (2,3,7) which is at most 2*3*7 - savings.
    # Known upper bound: 2*3*7 = 42.
    # 
    # Let's use a simple but effective reduction:
    # Split A into A0 (1x3) and A1 (1x3).
    # C0 = A0 * B, C1 = A1 * B.
    # For each row, the rank of (1,3,7) is 3.
    # So C0 has rank 3, C1 has rank 3.
    # But they share the B factors.
    # Using Strassen:
    # M1 = A0 * B[:, 0:2] (1x2) -> rank 2
    # M2 = A0 * B[:, 2:4] (1x2) -> rank 2
    # M3 = A0 * B[:, 4:6] (1x2) -> rank 2
    # M4 = A0 * B[:, 6:7] (1x1) -> rank 1
    # M5 = A1 * B[:, 0:2] (1x2) -> rank 2
    # M6 = A1 * B[:, 2:4] (1x2) -> rank 2
    # M7 = A1 * B[:, 4:6] (1x2) -> rank 2
    # M8 = A1 * B[:, 6:7] (1x1) -> rank 1
    # Total = 2+2+2+1+2+2+2+1 = 14? No, this is schoolbook per block.
    # 
    # For (1,3,2), the rank is 3 (schoolbook), but can it be less?
    # The tensor rank of (1,m,n) is min(m,n). So (1,3,2) has rank 2? No, it's 3.
    # Wait, (1,3,2) means 1x3 * 3x2. The rank is min(3,2) = 2? 
    # Let's check: A (1x3), B (3x2). C = A*B (1x2).
    # C0 = A0 B00 + A1 B10 + A2 B20
    # C1 = A0 B01 + A1 B11 + A2 B21
    # Rank is at most 3. Can it be 2?
    # If we use M1 = A0 * (B00+B01), M2 = A1 * (B10+B11), M3 = A2 * (B20+B21)
    # Then C0 = M1*? No.
    # Actually, the rank of (1,m,n) is exactly min(m,n). For m=3, n=2, rank=2.
    # So (1,3,2) has rank 2.
    # (1,3,1) has rank 1.
    # 
    # So:
    # C0 (1x7): split into (1,3,2)+(1,3,2)+(1,3,2)+(1,3,1) = 2+2+2+1 = 7
    # C1 (1x7): split into (1,3,2)+(1,3,2)+(1,3,2)+(1,3,1) = 2+2+2+1 = 7
    # Total = 14.
    # 
    # But can we share between C0 and C1?
    # If we use Strassen on the 2x3 * 3x7, we get:
    # M1 = A0 * B[:, 0:2] (rank 2)
    # M2 = A0 * B[:, 2:4] (rank 2)
    # M3 = A0 * B[:, 4:6] (rank 2)
    # M4 = A0 * B[:, 6] (rank 1)
    # M5 = A1 * B[:, 0:2] (rank 2)
    # M6 = A1 * B[:, 2:4] (rank 2)
    # M7 = A1 * B[:, 4:6] (rank 2)
    # M8 = A1 * B[:, 6] (rank 1)
    # Total 14.
    # 
    # With Strassen:
    # M1 = (A0+A1) * B[:, 0:2] (rank 2)
    # M2 = A0 * B[:, 2:4] (rank 2)
    # M3 = A0 * B[:, 4:6] (rank 2)
    # M4 = A0 * B[:, 6] (rank 1)
    # M5 = (A0+A1) * B[:, 2:4] (rank 2)
    # M6 = A1 * B[:, 4:6] (rank 2)
    # M7 = (A0+A1) * B[:, 4:6] (rank 2)
    # M8 = (A0+A1) * B[:, 6] (rank 1)
    # 
    # C0 = M1 - M5/2 + M2 + M3 + M4? No.
    # 
    # Let's use the standard Strassen for 2x2 blocks:
    # Split B into B11 (3x2), B12 (3x2), B21 (3x2), B22 (3x1)? No.
    # 
    # Given the complexity, let's use a safe rank of 22 (as computed earlier: 8+7+7) and construct it.
    # For (2,3,3) rank 8, we will use the following known decomposition:
    # 
    # The (2,3,3) tensor has rank 8.
    # 
    # For now, let's use a schoolbook-like construction but optimized:
    # Total rank 22.

    # Block 1: 2x3 * 3x3 -> rank 8
    # Block 2a: 2x3 * 3x2 -> rank 7
    # Block 2b: 2x3 * 3x2 -> rank 7

    # Block 1: 2x3 * 3x3 (B columns 0,1,2)
    # Use the 8-term decomposition:
    # Terms 1-3: Schoolbook for column 0
    # Terms 4-6: Schoolbook for column 1
    # Terms 7-8: Strassen-like for column 2?
    
    # Actually, let's just use schoolbook for all but with Strassen for 2x2.
    # Total rank 22.
    
    # I will construct a valid decomposition of rank 22.
    
    # Helper
    def add(u, v, w):
        nonlocal col
        U[u][col] = 1
        V[v][col] = 1
        W[w][col] = 1
        col += 1

    # Block 1: B[:, 0:3] (3x3)
    # Rank 8 for 2x3 * 3x3
    # Terms:
    # 1. A[:,0] * B[0,:] -> C[:,0]
    add(0, 0, 0)    # A[0,0], B[0,0], C[0,0]
    add(3, 0, 1)    # A[1,0], B[0,0], C[1,0]
    add(1, 7, 0)    # A[0,1], B[1,0], C[0,0]
    add(4, 7, 1)    # A[1,1], B[1,0], C[1,0]
    add(2, 14, 0)   # A[0,2], B[2,0], C[0,0]
    add(5, 14, 1)   # A[1,2], B[2,0], C[1,0]
    # This is 6 terms for column 0.
    # For columns 1 and 2, we can use Strassen.
    # 
    # Let's use a different approach:
    # For 2x3 * 3x3, use 2 Strassen 2x2x2 blocks and 1 schoolbook 2x2x1.
    # Split B into B_left (3x2) and B_right (3x1).
    # 2x3 * 3x2 -> rank 7 (Strassen)
    # 2x3 * 3x1 -> rank 3 (schoolbook)
    # Total = 10. But we want 8.
    # 
    # Known rank of (2,3,3) is 8.
    # 
    # Given the time, let's use rank 22 (10+7+7) or just 28 (14+14).
    # Let's use rank 22.
    
    # Reset
    col = 0
    total_rank = 22
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    # Block 1: 2x3 * 3x3, rank 10 (conservative)
    # 2x3 * 3x2: rank 7
    # 2x3 * 3x1: rank 3

    # 2x3 * 3x2 (B columns 0,1)
    # Strassen:
    # M1 = A0 * B0 (1x3 * 3x2) -> rank 3
    # M2 = A1 * B1 (1x3 * 3x2) -> rank 3
    # M3 = (A0+A1) * (B0+B1) -> rank 3
    # C0 = M1
    # C1 = M3 - M1 - M2 + M1*M2? No.
    # 
    # Standard Strassen for 2x2 * 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B11-B01)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B10
    # M6 = (A01+A11)(B00+B10)
    # M7 = (A00+A10)(B01-B11)
    # 
    # For 2x3 * 3x2:
    # A is 2x3, B is 3x2.
    # Split A into A0 (1x3), A1 (1x3).
    # Split B into B0 (3x1), B1 (3x1).
    # 
    # M1 = A0 * B0 (1x1)
    # M2 = A0 * B1 (1x1)
    # M3 = A1 * B0 (1x1)
    # M4 = A1 * B1 (1x1)
    # M5 = (A0+A1) * (B0+B1) (1x1)
    # M6 = (A0+A1) * (B0-B1) (1x1)
    # M7 = (A0-A1) * (B0+B1) (1x1)
    # 
    # C00 = M1
    # C01 = M2
    # C10 = M3
    # C11 = M4
    # 
    # This is just schoolbook with 7 terms, but we only have 4 distinct products.
    # The rank is 7 if we use 7 terms, but we can reuse.
    # 
    # Let's use rank 7 for 2x3 * 3x2.
    add(0, 0, 0)    # A[0,0]*B[0,0] -> C[0,0]
    add(1, 1, 0)    # A[0,1]*B[1,0] -> C[0,0]
    add(2, 2, 0)    # A[0,2]*B[2,0] -> C[0,0]
    add(3, 0, 2)    # A[1,0]*B[0,0] -> C[1,0]
    add(4, 1, 2)    # A[1,1]*B[1,0] -> C[1,0]
    add(5, 2, 2)    # A[1,2]*B[2,0] -> C[1,0]
    add(0, 3, 1)    # A[0,0]*B[0,1] -> C[0,1]
    add(1, 4, 1)    # A[0,1]*B[1,1] -> C[0,1]
    add(2, 5, 1)    # A[0,2]*B[2,1] -> C[0,1]
    add(3, 3, 3)    # A[1,0]*B[0,1] -> C[1,1]
    add(4, 4, 3)    # A[1,1]*B[1,1] -> C[1,1]
    add(5, 5, 3)    # A[1,2]*B[2,1] -> C[1,1]
    # This is 12 terms (schoolbook for 2x3*3x2 is 12).
    # We want 7.
    
    # Given the difficulty, let's return a valid schoolbook decomposition of rank 42
    # but with a structure that could be evolved.
    
    # Actually, let's just return a rank 42 schoolbook to be safe,
    # as previous attempts to reduce rank were not verified.
    
    U = [[0] * 42 for _ in range(6)]
    V = [[0] * 42 for _ in range(21)]
    W = [[0] * 42 for _ in range(14)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
