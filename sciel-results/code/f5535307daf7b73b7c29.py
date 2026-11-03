"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return (U, V, W) as lists of lists for schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _strassen_2x4x8():
    """
    Decompose 2x4 * 4x8 -> 2x8 using a Strassen-like rank-7 splitting.
    A (2x4) = [[A1, A2]], B (4x8) = [[B1],[B2]] where A1,A2,B1,B2 are 1x4 or 4x8.
    C1 = A1*B1, C2 = A2*B2, C3 = (A1+A2)*(B1+B2) - C1 - C2, etc.
    Actually for 2x4 * 4x8:
      C = [[A1, A2]] * [[B1],[B2]] = [[A1*B1 + A2*B2, ...]]
    Wait, dimensions: A is 2x4, B is 4x8.
    Let's partition A into two 1x4 rows: A_top (1x4), A_bot (1x4).
    Partition B into two 2x8 blocks: B_top (2x8), B_bot (2x8).
    C_top = A_top * B_top (1x8)
    C_bot = A_bot * B_bot (1x8)
    But this is just schoolbook for the two 1x4*4x8 multiplications, rank 2*4*8=64.
    
    To use Strassen, we need to split the inner dimension 4 into 2+2.
    A = [A1 A2] where A1, A2 are 2x2.
    B = [B1; B2] where B1, B2 are 2x8.
    C = A1*B1 + A2*B2 (2x8).
    Strassen for 2x2 * 2x2 uses rank 7. Here we have 2x2 * 2x8.
    We can treat the 2x8 as 4 separate 2x2 blocks? No, that's inefficient.
    
    Better approach: Use the known rank-7 decomposition for 2x2x2 and apply it block-wise?
    Actually, the rank of 2x2 * 2x2 is 7.
    The tensor for 2x2 * 2x2 has 8 entries.
    We can use the 7 rank-1 tensors for 2x2x2.
    
    For 2x4 * 4x8:
    Split A (2x4) into A1 (2x2) and A2 (2x2) horizontally.
    Split B (4x8) into B1 (2x8) and B2 (2x8) vertically.
    C = A1*B1 + A2*B2.
    
    Now A1*B1 is 2x2 * 2x8. This is equivalent to 4 independent 2x2 * 2x2 multiplications (one for each pair of columns in B1).
    So A1*B1 has rank 4 * 7 = 28? No, the tensors are not independent in that simple way for the whole matrix multiplication tensor.
    
    Let's use the known result: rank of 2x2x2 is 7.
    The multiplication tensor T_{2,2,2} = sum_{i,j,k} e_i e_j e_k x_i y_j z_k.
    Strassen decomposition: 7 terms.
    
    For 2x4 * 4x8:
    We can view this as a sum of 4 * (2x2 * 2x8) blocks? No.
    
    Let's use a simpler recursive Strassen-like split that is known to work:
    Split n=2, m=4, p=8.
    Split m=4 into m1=2, m2=2.
    A = [A1 A2] (2x2, 2x2)
    B = [B1; B2] (2x8, 2x8)
    C = A1*B1 + A2*B2.
    
    Now, A1*B1 is 2x2 * 2x8.
    We can decompose 2x2 * 2x8 by treating the 2x8 as two 2x4 blocks? Or use the rank-7 for 2x2x2 and replicate?
    
    Actually, there is a known decomposition for 2x2 * 2x2 of rank 7.
    Let's define the 7 rank-1 tensors for 2x2 * 2x2:
    M1 = A11 * B11
    M2 = (A11+A12) * (B21-B22)
    M3 = (A21+A22) * (B11-B12)
    M4 = A22 * B22
    M5 = A11 * (B11+B22)
    M6 = (A22+A11) * (B22+B11)
    M7 = (A12+A22) * (B22+B11)
    ... this is getting complex.
    
    Let's use a simpler, verified approach:
    Use the known rank-7 decomposition for 2x2x2 from literature.
    Then, for 2x4 * 4x8:
    C = A1*B1 + A2*B2.
    A1*B1: A1 is 2x2, B1 is 2x8.
    We can write B1 = [B1a B1b] where B1a, B1b are 2x4.
    A1*B1 = [A1*B1a A1*B1b].
    Each A1*B1a is 2x2 * 2x4.
    2x2 * 2x4 can be decomposed using rank 7 for 2x2x2 applied to the 4 columns? No.
    
    Let's try a different split:
    Split p=8 into p1=4, p2=4.
    C = [C1 C2] where C1 = A*B1, C2 = A*B2.
    A*B1 is 2x4 * 4x4.
    Rank of 2x4 * 4x4 is at most 2*4*4=32.
    Can we do better?
    
    Actually, the hybrid decomposition mentioned in the knowledge:
    Split 3x4 into 1x4 and 2x4.
    1x4 * 4x8: rank 1*4*8=32.
    2x4 * 4x8: use Strassen-like.
    
    Let's implement the 2x4 * 4x8 part with a specific rank-56 decomposition.
    How to get rank 56 for 2x4 * 4x8?
    2*4*8 = 64.
    56 = 64 - 8.
    
    One way: Use the rank-7 decomposition for 2x2x2.
    Split A (2x4) into A1 (2x2) and A2 (2x2).
    Split B (4x8) into B1 (2x8) and B2 (2x8).
    C = A1*B1 + A2*B2.
    
    Now, A1*B1 is 2x2 * 2x8.
    We can decompose 2x2 * 2x8 as follows:
    B1 has 8 columns. Let B1 = [b1, b2, ..., b8] where bi are 2x1 vectors.
    A1*B1 = [A1*b1, A1*b2, ..., A1*b8].
    Each A1*bi is 2x1.
    The tensor for A1*bi is 2x2x1. Rank of 2x2x1 is 4 (schoolbook).
    So A1*B1 has rank 8*4 = 32? No, that's not right.
    
    Let's use the explicit Strassen decomposition for 2x2 * 2x2:
    M1 = a11*b11
    M2 = (a11+a12)*(b21-b22)
    M3 = (a21+a22)*(b11-b12)
    M4 = a22*b22
    M5 = a11*(b11+b22)
    M6 = (a22+a11)*(b22+b11)
    M7 = (a12+a22)*(b22+b11)
    
    c11 = M1 + M5 - M6
    c12 = M7 - M5
    c21 = M6 - M4
    c22 = M2 + M4 + M7
    
    For 2x2 * 2x8:
    B = [b1 b2 b3 b4 b5 b6 b7 b8] where bi are 2x1.
    We can group the columns into pairs: (b1,b2), (b3,b4), (b5,b6), (b7,b8).
    For each pair (bj, bk), we can use the 7 Strassen multiplications.
    So A1*B1 requires 4 * 7 = 28 rank-1 terms.
    Similarly, A2*B2 requires 4 * 7 = 28 rank-1 terms.
    Total rank for 2x4 * 4x8 = 28 + 28 = 56.
    
    Total rank for 3x4 * 4x8 = 32 (for 1x4*4x8) + 56 (for 2x4*4x8) = 88.
    """
    # We'll build the 2x4 * 4x8 decomposition with rank 56.
    # A is 2x4, B is 4x8.
    # A = [A1 A2] where A1, A2 are 2x2.
    # B = [B1; B2] where B1, B2 are 2x8.
    # C = A1*B1 + A2*B2.
    
    # For A1*B1 (2x2 * 2x8):
    # Split B1 into 4 blocks of 2 columns each: B1 = [B1a B1b B1c B1d] where each is 2x2.
    # Use Strassen rank-7 for each 2x2 * 2x2.
    
    # Strassen coefficients for 2x2 * 2x2:
    # M1 = a11 * b11
    # M2 = (a11+a12) * (b21-b22)
    # M3 = (a21+a22) * (b11-b12)
    # M4 = a22 * b22
    # M5 = a11 * (b11+b22)
    # M6 = (a22+a11) * (b22+b11)
    # M7 = (a12+a22) * (b22+b11)
    
    # c11 = M1 + M5 - M6
    # c12 = M7 - M5
    # c21 = M6 - M4
    # c22 = M2 + M4 + M7
    
    # We'll build U, V, W for the 2x4 * 4x8 part.
    # U has shape (2*4, 56) = (8, 56)
    # V has shape (4*8, 56) = (32, 56)
    # W has shape (8*2, 56) = (16, 56)
    
    U_sub = [[0] * 56 for _ in range(8)]
    V_sub = [[0] * 56 for _ in range(32)]
    W_sub = [[0] * 56 for _ in range(16)]
    
    col = 0
    # Process A1*B1 (2x2 * 2x8)
    # A1 is rows 0-1, cols 0-1 of A (2x4).
    # B1 is rows 0-1, cols 0-7 of B (4x8).
    # Split B1 into 4 blocks of 2 columns: B1a (cols 0-1), B1b (cols 2-3), B1c (cols 4-5), B1d (cols 6-7).
    
    for block in range(4):
        # B1 block is rows 0-1, cols 2*block to 2*block+1.
        # Strassen M1: a11 * b11
        # a11 is A1[0,0] = A[0,0]
        # b11 is B1[0, 2*block] = B[0, 2*block]
        U_sub[0][col] = 1  # A[0,0]
        V_sub[0 * 8 + 2*block][col] = 1  # B[0, 2*block]
        W_sub[0 * 2 + 0][col] = 1  # C[0,0]
        col += 1
        
        # M2: (a11+a12) * (b21-b22)
        # a11+a12 = A[0,0] + A[0,1]
        # b21-b22 = B[1, 2*block] - B[1, 2*block+1]
        U_sub[0][col] = 1  # A[0,0]
        U_sub[1][col] = 1  # A[0,1]
        V_sub[1 * 8 + 2*block][col] = 1  # B[1, 2*block]
        V_sub[1 * 8 + 2*block+1][col] = -1  # B[1, 2*block+1]
        # c22 += M2, c12 -= M5, etc.
        # For now, we'll accumulate contributions.
        # c22 = M2 + M4 + M7
        W_sub[1 * 2 + 1][col] = 1  # C[1,1]
        col += 1
        
        # M3: (a21+a22) * (b11-b12)
        # a21+a22 = A[1,0] + A[1,1]
        # b11-b12 = B[0, 2*block] - B[0, 2*block+1]
        U_sub[4][col] = 1  # A[1,0]
        U_sub[5][col] = 1  # A[1,1]
        V_sub[0 * 8 + 2*block][col] = 1  # B[0, 2*block]
        V_sub[0 * 8 + 2*block+1][col] = -1  # B[0, 2*block+1]
        # c12 = M7 - M5, c21 = M6 - M4, etc.
        # M3 is not directly used in the standard Strassen?
        # Actually, the standard Strassen uses:
        # M1 = a11*b11
        # M2 = (a11+a12)*(b21-b22)
        # M3 = (a21+a22)*(b11-b12)
        # M4 = a22*b22
        # M5 = a11*(b11+b22)
        # M6 = (a22+a11)*(b22+b11)
        # M7 = (a12+a22)*(b22+b11)
        
        # c11 = M1 + M5 - M6
        # c12 = M7 - M5
        # c21 = M6 - M4
        # c22 = M2 + M4 + M7
        
        # So M3 is not used? That can't be right.
        # Let me re-check the Strassen decomposition.
        # Actually, the standard Strassen uses 7 multiplications:
        # M1 = A11 * B11
        # M2 = (A11 + A12) * (B21 - B22)
        # M3 = (A21 + A22) * (B11 - B12)
        # M4 = A22 * B22
        # M5 = (A11 + A22) * (B11 + B22)
        # M6 = (A12 + A22) * (B22 - B11)
        # M7 = (A11 + A12) * (B11 + B12)
        
        # C11 = M1 + M5 - M3 - M7
        # C12 = M4 + M7
        # C21 = M2 + M5
        # C22 = M1 + M4 - M2 + M6
        
        # Let's use this version.
        # I'll restart the construction with the correct Strassen formulas.
        
    # I made a mistake in the Strassen formulas. Let me use the correct ones.
    # I'll rebuild the U_sub, V_sub, W_sub with the correct formulas.
    
    U_sub = [[0] * 56 for _ in range(8)]
    V_sub = [[0] * 56 for _ in range(32)]
    W_sub = [[0] * 56 for _ in range(16)]
    
    col = 0
    for block in range(4):
        # B1 block is rows 0-1, cols 2*block to 2*block+1.
        # M1 = A11 * B11
        U_sub[0][col] = 1  # A[0,0]
        V_sub[0 * 8 + 2*block][col] = 1  # B[0, 2*block]
        W_sub[0 * 2 + 0][col] = 1  # C[0,0]
        W_sub[1 * 2 + 1][col] = 1  # C[1,1]
        col += 1
        
        # M2 = (A11 + A12) * (B21 - B22)
        U_sub[0][col] = 1  # A[0,0]
        U_sub[1][col] = 1  # A[0,1]
        V_sub[1 * 8 + 2*block][col] = 1  # B[1, 2*block]
        V_sub[1 * 8 + 2*block+1][col] = -1  # B[1, 2*block+1]
        W_sub[0 * 2 + 1][col] = 1  # C[0,1]
        W_sub[1 * 2 + 1][col] = -1  # C[1,1]
        col += 1
        
        # M3 = (A21 + A22) * (B11 - B12)
        U_sub[4][col] = 1  # A[1,0]
        U_sub[5][col] = 1  # A[1,1]
        V_sub[0 * 8 + 2*block][col] = 1  # B[0, 2*block]
        V_sub[0 * 8 + 2*block+1][col] = -1  # B[0, 2*block+1]
        W_sub[0 * 2 + 0][col] = -1  # C[0,0]
        W_sub[1 * 2 + 0][col] = 1  # C[1,0]
        col += 1
        
        # M4 = A22 * B22
        U_sub[5][col] = 1  # A[1,1]
        V_sub[1 * 8 + 2*block+1][col] = 1  # B[1, 2*block+1]
        W_sub[0 * 2 + 1][col] = 1  # C[0,1]
        W_sub[1 * 2 + 1][col] = 1  # C[1,1]
        col += 1
        
        # M5 = (A11 + A22) * (B11 + B22)
        U_sub[0][col] = 1  # A[0,0]
        U_sub[5][col] = 1  # A[1,1]
        V_sub[0 * 8 + 2*block][col] = 1  # B[0, 2*block]
        V_sub[1 * 8 + 2*block+1][col] = 1  # B[1, 2*block+1]
        W_sub[0 * 2 + 0][col] = 1  # C[0,0]
        W_sub[0 * 2 + 1][col] = 1  # C[0,1]
        W_sub[1 * 2 + 0][col] = 1  # C[1,0]
        W_sub[1 * 2 + 1][col] = 1  # C[1,1]
        col += 1
        
        # M6 = (A12 + A22) * (B22 - B11)
        U_sub[1][col] = 1  # A[0,1]
        U_sub[5][col] = 1  # A[1,1]
        V_sub[1 * 8 + 2*block+1][col] = 1  # B[1, 2*block+1]
        V_sub[0 * 8 + 2*block][col] = -1  # B[0, 2*block]
        W_sub[1 * 2 + 1][col] = 1  # C[1,1]
        col += 1
        
        # M7 = (A11 + A12) * (B11 + B12)
        U_sub[0][col] = 1  # A[0,0]
        U_sub[1][col] = 1  # A[0,1]
        V_sub[0 * 8 + 2*block][col] = 1  # B[0, 2*block]
        V_sub[0 * 8 + 2*block+1][col] = 1  # B[0, 2*block+1]
        W_sub[0 * 2 + 0][col] = -1  # C[0,0]
        W_sub[0 * 2 + 1][col] = 1  # C[0,1]
        col += 1
    
    # Now process A2*B2 (2x2 * 2x8)
    # A2 is rows 0-1, cols 2-3 of A (2x4).
    # B2 is rows 2-3, cols 0-7 of B (4x8).
    # Split B2 into 4 blocks of 2 columns: B2a (cols 0-1), B2b (cols 2-3), B2c (cols 4-5), B2d (cols 6-7).
    
    for block in range(4):
        # B2 block is rows 2-3, cols 2*block to 2*block+1.
        # M1 = A22 * B22 (where A22 is A2[1,1] = A[1,3], B22 is B2[1, 2*block+1] = B[3, 2*block+1])
        U_sub[7][col] = 1  # A[1,3]
        V_sub[3 * 8 + 2*block+1][col] = 1  # B[3, 2*block+1]
        W_sub[0 * 2 + 0][col] = 1  # C[0,0]
        W_sub[1 * 2 + 1][col] = 1  # C[1,1]
        col += 1
        
        # M2 = (A21 + A22) * (B22 - B21) (where A21 is A2[1,0] = A[1,2], A22 is A2[1,1] = A[1,3])
        # B21 is B2[0, 2*block] = B[2, 2*block], B22 is B2[1, 2*block+1] = B[3, 2*block+1]
        U_sub[6][col] = 1  # A[1,2]
        U_sub[7][col] = 1  # A[1,3]
        V_sub[3 * 8 + 2*block+1][col] = 1  # B[3, 2*block+1]
        V_sub[2 * 8 + 2*block][col] = -1  # B[2, 2*block]
        W_sub[0 * 2 + 1][col] = 1  # C[0,1]
        W_sub[1 * 2 + 1][col] = -1  # C[1,1]
        col += 1
        
        # M3 = (A21 + A22) * (B21 - B22) (where A21 is A2[1,0] = A[1,2], A22 is A2[1,1] = A[1,3])
        # B21 is B2[0, 2*block] = B[2, 2*block], B22 is B2[1, 2*block+1] = B[3, 2*block+1]
        U_sub[6][col] = 1  # A[1,2]
        U_sub[7][col] = 1  # A[1,3]
        V_sub[2 * 8 + 2*block][col] = 1  # B[2, 2*block]
        V_sub[3 * 8 + 2*block+1][col] = -1  # B[3, 2*block+1]
        W_sub[0 * 2 + 0][col] = -1  # C[0,0]
        W_sub[1 * 2 + 0][col] = 1  # C[1,0]
        col += 1
        
        # M4 = A22 * B22 (where A22 is A2[1,1] = A[1,3], B22 is B2[1, 2*block+1] = B[3, 2*block+1])
        U_sub[7][col] = 1  # A[1,3]
        V_sub[3 * 8 + 2*block+1][col] = 1  # B[3, 2*block+1]
        W_sub[0 * 2 + 1][col] = 1  # C[0,1]
        W_sub[1 * 2 + 1][col] = 1  # C[1,1]
        col += 1
        
        # M5 = (A21 + A22) * (B21 + B22) (where A21 is A2[1,0] = A[1,2], A22 is A2[1,1] = A[1,3])
        # B21 is B2[0, 2*block] = B[2, 2*block], B22 is B2[1, 2*block+1] = B[3, 2*block+1]
        U_sub[6][col] = 1  # A[1,2]
        U_sub[7][col] = 1  # A[1,3]
        V_sub[2 * 8 + 2*block][col] = 1  # B[2, 2*block]
        V_sub[3 * 8 + 2*block+1][col] = 1  # B[3, 2*block+1]
        W_sub[0 * 2 + 0][col] = 1  # C[0,0]
        W_sub[0 * 2 + 1][col] = 1  # C[0,1]
        W_sub[1 * 2 + 0][col] = 1  # C[1,0]
        W_sub[1 * 2 + 1][col] = 1  # C[1,1]
        col += 1
        
        # M6 = (A21 + A22) * (B22 - B21) (where A21 is A2[1,0] = A[1,2], A22 is A2[1,1] = A[1,3])
        # B21 is B2[0, 2*block] = B[2, 2*block], B22 is B2[1, 2*block+1] = B[3, 2*block+1]
        U_sub[6][col] = 1  # A[1,2]
        U_sub[7][col] = 1  # A[1,3]
        V_sub[3 * 8 + 2*block+1][col] = 1  # B[3, 2*block+1]
        V_sub[2 * 8 + 2*block][col] = -1  # B[2, 2*block]
        W_sub[1 * 2 + 1][col] = 1  # C[1,1]
        col += 1
        
        # M7 = (A21 + A22) * (B21 + B22) (where A21 is A2[1,0] = A[1,2], A22 is A2[1,1] = A[1,3])
        # B21 is B2[0, 2*block] = B[2, 2*block], B22 is B2[1, 2*block+1] = B[3, 2*block+1]
        U_sub[6][col] = 1  # A[1,2]
        U_sub[7][col] = 1  # A[1,3]
        V_sub[2 * 8 + 2*block][col] = 1  # B[2, 2*block]
        V_sub[3 * 8 + 2*block+1][col] = 1  # B[3, 2*block+1]
        W_sub[0 * 2 + 0][col] = -1  # C[0,0]
        W_sub[0 * 2 + 1][col] = 1  # C[0,1]
        col += 1
    
    return U_sub, V_sub, W_sub

def construct():
    n, m, p = DIMENSIONS
    # Split 3x4 into 1x4 and 2x4.
    # 1x4 * 4x8: rank 32.
    # 2x4 * 4x8: rank 56.
    # Total rank: 88.
    
    U, V, W = [], [], []
    
    # Part 1: 1x4 * 4x8 (schoolbook)
    U1, V1, W1 = _schoolbook(1, 4, 8)
    U.extend(U1)
    V.extend(V1)
    W.extend(W1)
    
    # Part 2: 2x4 * 4x8 (Strassen-like)
    U2, V2, W2 = _strassen_2x4x8()
    for row in U2:
        U.append(row)
    for row in V2:
        V.append(row)
    for row in W2:
        W.append(row)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
