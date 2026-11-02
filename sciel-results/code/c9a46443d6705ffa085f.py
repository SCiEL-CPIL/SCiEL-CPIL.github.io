"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from collections import defaultdict

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication."""
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    
    # We will use a decomposition based on Strassen for 2x2x2x2 blocks 
    # and schoolbook for 2x2x2x1 blocks.
    # A is 4x4, B is 4x5.
    # Split A into 2x2 blocks: A00, A01, A10, A11 (each 2x2)
    # Split B into 2x2 and 2x1 blocks:
    # B00 (2x2), B01 (2x1) -> Top 2 rows of B, cols 0-1 and 2
    # B10 (2x2), B11 (2x1) -> Bottom 2 rows of B, cols 0-1 and 2
    # Wait, B is 4x5.
    # Rows 0-1: B_top (2x5). Split into B00 (2x2) and B01 (2x3)?
    # If we split B into 2x2 and 2x3, the "2x1" part becomes 2x3.
    # Let's stick to 2x2 and 2x1 for the first 4 columns, and handle the 5th column separately?
    # Or split B into 2x2 (cols 0,1) and 2x3 (cols 2,3,4).
    # Then the "2x2x2x1" part becomes "2x2x2x3".
    # Rank of 2x2x2x3 is at most 11?
    # Let's use the 2x2 and 2x3 split.
    
    # Blocks of A:
    # A00 (0,0), A01 (0,2), A10 (2,0), A11 (2,2)
    
    # Blocks of B:
    # B00 (0,0) 2x2, B01 (0,2) 2x3
    # B10 (2,0) 2x2, B11 (2,2) 2x3
    
    # Products:
    # P1 = A00 * B00 (2x2 * 2x2) -> Rank 7
    # P2 = A01 * B10 (2x2 * 2x2) -> Rank 7
    # P3 = A10 * B00 (2x2 * 2x2) -> Rank 7
    # P4 = A11 * B10 (2x2 * 2x2) -> Rank 7
    
    # Q1 = A00 * B01 (2x2 * 2x3)
    # Q2 = A01 * B11 (2x2 * 2x3)
    # Q3 = A10 * B01 (2x2 * 2x3)
    # Q4 = A11 * B11 (2x2 * 2x3)
    
    # Rank of 2x2 * 2x3:
    # Can be decomposed into 2x2*2x2 + 2x2*2x1?
    # 2x2*2x2 is rank 7.
    # 2x2*2x1 is rank 4.
    # Total 11.
    
    # Total Rank = 4 * 7 + 4 * 11 = 28 + 44 = 72.
    # This is better than 80.
    
    # Let's verify the rank of 2x2*2x3.
    # C (2x3) = A (2x2) * B (2x3).
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C02 = A00 B02 + A01 B12
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    # C12 = A10 B02 + A11 B12
    
    # We can use Strassen for the 2x2 parts (cols 0,1) and schoolbook for col 2.
    # Strassen for 2x2*2x2 (cols 0,1 of B):
    # B' = B[:, 0:2]
    # Compute A * B' using Strassen (Rank 7).
    # Schoolbook for col 2:
    # B'' = B[:, 2:3]
    # Compute A * B'' using Schoolbook (Rank 4).
    # Total Rank 11.
    
    # So Total Rank = 4 * 7 + 4 * 11 = 72.
    
    def strassen_2x2x2x2():
        """Returns U, V, W for 2x2 * 2x2 -> 2x2."""
        # Indices: A(0,0), A(0,1), A(1,0), A(1,1)
        # B(0,0), B(0,1), B(1,0), B(1,1)
        # C(0,0), C(0,1), C(1,0), C(1,1)
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[1][0] = 1
        V[0][0] = 1; V[1][0] = 1
        W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1
        
        # M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = 1
        
        # M3 = A00(B01-B11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4 = A11(B10-B00)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[1][4] = 1; W[3][4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[2][5] = 1; W[3][5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[2][6] = 1
        
        return U, V, W

    def schoolbook_2x2x2x1():
        """Returns U, V, W for 2x2 * 2x1 -> 2x1."""
        # A(0,0), A(0,1), A(1,0), A(1,1)
        # B(0,0), B(1,0)
        # C(0,0), C(1,0)
        
        U = [[0]*4 for _ in range(4)]
        V = [[0]*4 for _ in range(2)]
        W = [[0]*4 for _ in range(2)]
        
        # C0 = A00 B0 + A01 B1
        U[0][0] = 1
        V[0][0] = 1
        W[0][0] = 1
        
        U[1][1] = 1
        V[1][1] = 1
        W[0][1] = 1
        
        # C1 = A10 B0 + A11 B1
        U[2][2] = 1
        V[0][2] = 1
        W[1][2] = 1
        
        U[3][3] = 1
        V[1][3] = 1
        W[1][3] = 1
        
        return U, V, W

    # Construct the full 4x4 * 4x5 decomposition
    # Total Rank = 4 * 7 + 4 * (7 + 4) = 28 + 44 = 72
    
    rank = 72
    U = [[0] * rank for _ in range(n * m)] # 16 x 72
    V = [[0] * rank for _ in range(m * p)] # 20 x 72
    W = [[0] * rank for _ in range(p * n)] # 20 x 72
    
    # Helper to add a sub-decomposition
    def add_decomp(u_sub, v_sub, w_sub, a_offset, b_offset, c_offset, rank_offset):
        """
        u_sub: 4xR (2x2 A block)
        v_sub: 2xR or 2xR (2x1 or 2x2 B block)
        w_sub: 2xR or 4xR (2x1 or 2x2 C block)
        a_offset: index in A (0, 2, 8, 10)
        b_offset: index in B
        c_offset: index in C
        rank_offset: index in rank
        """
        for i in range(len(u_sub)):
            for r in range(len(u_sub[0])):
                U[a_offset + i][rank_offset + r] += u_sub[i][r]
        
        for i in range(len(v_sub)):
            for r in range(len(v_sub[0])):
                V[b_offset + i][rank_offset + r] += v_sub[i][r]
        
        for i in range(len(w_sub)):
            for r in range(len(w_sub[0])):
                W[c_offset + i][rank_offset + r] += w_sub[i][r]

    # Strassen 2x2x2x2
    sU, sV, sW = strassen_2x2x2x2()
    
    # Schoolbook 2x2x2x1
    sbU, sbV, sbW = schoolbook_2x2x2x1()
    
    # Block 1: A00 * B00 (2x2 * 2x2) -> C00 (2x2)
    # A00 is A[0:2, 0:2] -> offset 0
    # B00 is B[0:2, 0:2] -> offset 0
    # C00 is C[0:2, 0:2] -> offset 0
    add_decomp(sU, sV, sW, 0, 0, 0, 0)
    
    # Block 2: A01 * B10 (2x2 * 2x2) -> C01 (2x2)
    # A01 is A[0:2, 2:4] -> offset 2
    # B10 is B[2:4, 0:2] -> offset 10 (2*5 + 0)
    # C01 is C[0:2, 2:4] -> offset 2 (0*4 + 2)
    add_decomp(sU, sV, sW, 2, 10, 2, 7)
    
    # Block 3: A10 * B00 (2x2 * 2x2) -> C10 (2x2)
    # A10 is A[2:4, 0:2] -> offset 8
    # B00 is B[0:2, 0:2] -> offset 0
    # C10 is C[2:4, 0:2] -> offset 8 (2*4 + 0)
    add_decomp(sU, sV, sW, 8, 0, 8, 14)
    
    # Block 4: A11 * B10 (2x2 * 2x2) -> C11 (2x2)
    # A11 is A[2:4, 2:4] -> offset 10
    # B10 is B[2:4, 0:2] -> offset 10
    # C11 is C[2:4, 2:4] -> offset 10 (2*4 + 2)
    add_decomp(sU, sV, sW, 10, 10, 10, 21)
    
    # Now the 2x2 * 2x3 parts
    # We split B01 (2x3) into B01a (2x2) and B01b (2x1)
    # B01a is B[0:2, 2:4] -> offset 2
    # B01b is B[0:2, 4:5] -> offset 4
    
    # Q1 = A00 * B01
    # A00 offset 0
    # B01a offset 2, B01b offset 4
    # C01 is C[0:2, 2:4] (from B01a) and C[0:2, 4:5] (from B01b)
    # Wait, C01 in the 2x2 block notation was C[0:2, 2:4].
    # But here we are computing the contribution to C[0:2, 2:5].
    # C[0:2, 2:4] is a 2x2 block. C[0:2, 4:5] is a 2x1 block.
    
    # Q1a = A00 * B01a (2x2 * 2x2)
    add_decomp(sU, sV, sW, 0, 2, 2, 28)
    
    # Q1b = A00 * B01b (2x2 * 2x1)
    add_decomp(sbU, sbV, sbW, 0, 4, 4, 35)
    
    # Q2 = A01 * B11
    # A01 offset 2
    # B11 is B[2:4, 2:5]. B11a is B[2:4, 2:4] -> offset 12. B11b is B[2:4, 4:5] -> offset 14.
    # C11 is C[2:4, 2:5]. C[2:4, 2:4] offset 10. C[2:4, 4:5] offset 12.
    
    # Q2a = A01 * B11a
    add_decomp(sU, sV, sW, 2, 12, 10, 39)
    
    # Q2b = A01 * B11b
    add_decomp(sbU, sbV, sbW, 2, 14, 12, 46)
    
    # Q3 = A10 * B01
    # A10 offset 8
    # B01a offset 2, B01b offset 4
    # C10 is C[2:4, 2:5]. C[2:4, 2:4] offset 10. C[2:4, 4:5] offset 12.
    
    # Q3a = A10 * B01a
    add_decomp(sU, sV, sW, 8, 2, 10, 50)
    
    # Q3b = A10 * B01b
    add_decomp(sbU, sbV, sbW, 8, 4, 12, 57)
    
    # Q4 = A11 * B11
    # A11 offset 10
    # B11a offset 12, B11b offset 14
    # C11 is C[2:4, 2:5]. C[2:4, 2:4] offset 10. C[2:4, 4:5] offset 12.
    
    # Q4a = A11 * B11a
    add_decomp(sU, sV, sW, 10, 12, 10, 61)
    
    # Q4b = A11 * B11b
    add_decomp(sbU, sbV, sbW, 10, 14, 12, 68)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
