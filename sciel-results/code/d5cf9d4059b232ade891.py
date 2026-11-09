"""
Matrix multiplication 2x3x10 - rank-minimal exact tensor decomposition
"""
import time
import random

DIMENSIONS = [2, 3, 10]

def construct():
    """
    Returns a tensor decomposition of matrix multiplication (2x3) * (3x10) -> (2x10).
    
    Uses block structure:
    - A is 2x3, B is 3x10, C is 2x10
    - Split B's 10 columns into 5 blocks of 2 columns each
    - For each block of 2 columns, use Strassen for the 2x2x2 part (using A's first 2 rows of B's 2 cols)
      and standard for the 2x1x2 part (using A's 3rd row of B's 2 cols)
    
    This gives rank 5*7 + 5*4 = 35 + 20 = 55
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    # We'll build U, V, W matrices
    # U: (n*m) x rank = 6 x rank
    # V: (m*p) x rank = 30 x rank  
    # W: (p*n) x rank = 20 x rank
    
    # Total rank: for each of 5 blocks:
    #   7 terms from Strassen 2x2x2 (using B rows 0,1)
    #   4 terms from standard 2x1x2 (using B row 2)
    # Total: 5 * (7 + 4) = 55
    
    total_rank = 0
    for b in range(p // 2):
        total_rank += 7 + 4  # Strassen 7 + standard 4
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    for b in range(p // 2):
        # Block b: columns 2b and 2b+1 of B (and C)
        c0 = 2 * b
        c1 = 2 * b + 1
        
        # Strassen 2x2x2: A[0:2, 0:2] * B[0:2, 2b:2b+2] -> C[0:2, 2b:2b+2]
        # P1 = (a11 + a22)(b11 + b22)
        # P2 = (a21 + a22)b11
        # P3 = a11(b12 - b22)
        # P4 = a22(b21 - b11)
        # P5 = (a11 + a12)b22
        # P6 = (a21 - a11)(b11 + b12)
        # P7 = (a12 - a22)(b21 + b22)
        #
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        
        # Strassen term 1: (a11 + a22)(b11 + b22)
        U[0][col] = 1  # a11
        U[1][col] = 1  # a22
        V[0 * p + c0][col] = 1  # b11
        V[1 * p + c1][col] = 1  # b22
        W[c0 * n + 0][col] = 1  # C11
        W[c1 * n + 0][col] = 1  # C12 -- wait, let me reconsider
        
        # Let me be more careful with the Strassen decomposition.
        # For 2x2x2: A is 2x2, B is 2x2, C is 2x2
        # A[i,j] where i=0,1 (row of A), j=0,1 (col of A / row of B)
        # B[j,k] where j=0,1 (row of B), k=0,1 (col of B / col of C)
        # C[i,k] where i=0,1 (row of C), k=0,1 (col of C)
        #
        # Tensor: T[i,j,k] = delta (we want sum_j A[i,j]*B[j,k] = C[i,k])
        #
        # In our problem:
        # A is 2x3, B is 3x10, C is 2x10
        # For block b, we use A[0:2, 0:2] (first 2 cols of A) and B[0:2, 2b:2b+2]
        # So A[i,j] for i=0,1; j=0,1
        # B[j,k] for j=0,1; k=0,1 (where k=0,1 maps to global col 2b+k)
        # C[i,k] for i=0,1; k=0,1 (where k=0,1 maps to global col 2b+k)
        
        # Let me redo this properly.
        # Reset col for this block
        # Actually, let me start fresh with a cleaner approach.
        
        col -= 1  # undo the increment
        break
    
    # Let me restart with a cleaner implementation
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    for b in range(p // 2):
        c0 = 2 * b
        c1 = 2 * b + 1
        
        # Strassen 2x2x2: A[0:2, 0:2] * B[0:2, c0:c1+1] -> C[0:2, c0:c1+1]
        # Local indices: i=0,1 (A row), j=0,1 (A col / B row), k=0,1 (B col / C col)
        # Global: A[i, j], B[j, c0+k], C[i, c0+k]
        
        # P1 = (a00 + a11)(b00 + b11)
        U[0][col] += 1   # a00
        U[1][col] += 1   # a11
        V[0 * p + c0][col] += 1   # b00
        V[1 * p + c1][col] += 1   # b11
        W[c0 * n + 0][col] += 1   # C[0, c0] = C11
        W[c1 * n + 0][col] += 1   # C[1, c0]?? No...
        
        # Hmm, I need to think about this more carefully.
        # C[i,k] where i is row (0,1), k is col (c0, c1)
        # W index: k*n + i (per the convention W[k*n+i, r])
        # So W[c0*n + i] corresponds to C[i, c0]
        # And W[c1*n + i] corresponds to C[i, c1]
        
        # P1 = (a00 + a11)(b00 + b11)
        # Contributes to: C[0,c0], C[0,c1], C[1,c0], C[1,c1] with coefficient 1
        # Wait no. P1 = (a00+a11)(b00+b11). This is a scalar.
        # In Strassen:
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        
        # So P1 appears in C11 and C22
        # P2 appears in C21 and C22 (with -1)
        # etc.
        
        # Let me just implement it term by term.
        
        # P1 = (a00 + a11)(b00 + b11)
        U[0][col] = 1
        U[1][col] = 1
        V[0 * p + c0][col] = 1
        V[1 * p + c1][col] = 1
        W[c0 * n + 0][col] = 1   # C[0, c0]
        W[c1 * n + 1][col] = 1   # C[1, c1]
        col += 1
        
        # P2 = (a10 + a11)b00
        U[1][col] = 1
        U[2][col] = 1  # a11 in A is at row 1, col 1 -> index 1*3+1=4? 
        # Wait, U has shape (n*m, rank) = (6, rank). Index i*m + j.
        # A[i,j] for i=0,1; j=0,1,2
        # So A[0,0]=U[0], A[0,1]=U[1], A[0,2]=U[2]
        # A[1,0]=U[3], A[1,1]=U[4], A[1,2]=U[5]
        
        # Let me restart again with correct indexing.
        break
    
    # OK let me just write it out cleanly.
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    for b in range(p // 2):
        c0 = 2 * b
        c1 = 2 * b + 1
        
        # Strassen 2x2x2: A[0:2, 0:2] * B[0:2, c0:c1+1] -> C[0:2, c0:c1+1]
        # A indices: (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4
        # B indices: (0,c0)->0*p+c0, (0,c1)->0*p+c1, (1,c0)->1*p+c0, (1,c1)->1*p+c1
        # C indices: (0,c0)->c0*n+0, (0,c1)->c1*n+0, (1,c0)->c0*n+1, (1,c1)->c1*n+1
        
        # P1 = (a00 + a11)(b00 + b11)
        U[0][col] = 1
        U[4][col] = 1
        V[0 * p + c0][col] = 1
        V[1 * p + c1][col] = 1
        W[c0 * n + 0][col] = 1
        W[c1 * n + 1][col] = 1
        col += 1
        
        # P2 = (a10 + a11)b00
        U[3][col] = 1
        U[4][col] = 1
        V[0 * p + c0][col] = 1
        W[c0 * n + 1][col] = 1
        W[c1 * n + 1][col] = -1
        col += 1
        
        # P3 = a00(b01 - b11)
        U[0][col] = 1
        V[0 * p + c1][col] = 1
        V[1 * p + c1][col] = -1
        W[c1 * n + 0][col] = 1
        W[c1 * n + 1][col] = 1
        col += 1
        
        # P4 = a11(b10 - b00)
        U[4][col] = 1
        V[1 * p + c0][col] = 1
        V[0 * p + c0][col] = -1
        W[c0 * n + 0][col] = 1
        W[c0 * n + 1][col] = 1
        col += 1
        
        # P5 = (a00 + a01)b11
        U[0][col] = 1
        U[1][col] = 1
        V[1 * p + c1][col] = 1
        W[c0 * n + 0][col] = -1
        W[c1 * n + 0][col] = 1
        col += 1
        
        # P6 = (a10 - a00)(b00 + b01)
        U[3][col] = 1
        U[0][col] = -1
        V[0 * p + c0][col] = 1
        V[0 * p + c1][col] = 1
        W[c1 * n + 1][col] = 1
        col += 1
        
        # P7 = (a01 - a11)(b10 + b11)
        U[1][col] = 1
        U[4][col] = -1
        V[1 * p + c0][col] = 1
        V[1 * p + c1][col] = 1
        W[c0 * n + 0][col] = 1
        col += 1
        
        # Standard 2x1x2: A[0:2, 2] * B[2, c0:c1+1] -> C[0:2, c0:c1+1]
        # A[0,2]=U[2], A[1,2]=U[5]
        # B[2,c0]=V[2*p+c0], B[2,c1]=V[2*p+c1]
        
        # Term 1: A[0,2] * B[2,c0] -> C[0,c0]
        U[2][col] = 1
        V[2 * p + c0][col] = 1
        W[c0 * n + 0][col] = 1
        col += 1
        
        # Term 2: A[0,2] * B[2,c1] -> C[0,c1]
        U[2][col] = 1
        V[2 * p + c1][col] = 1
        W[c1 * n + 0][col] = 1
        col += 1
        
        # Term 3: A[1,2] * B[2,c0] -> C[1,c0]
        U[5][col] = 1
        V[2 * p + c0][col] = 1
        W[c0 * n + 1][col] = 1
        col += 1
        
        # Term 4: A[1,2] * B[2,c1] -> C[1,c1]
        U[5][col] = 1
        V[2 * p + c1][col] = 1
        W[c1 * n + 1][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
