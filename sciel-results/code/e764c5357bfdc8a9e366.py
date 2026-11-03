import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 36 decomposition
    # Split A (3x4) into AL (3x2) and AR (3x2)
    # Split B (4x4) into B_LL (2x2), B_LR (2x2), B_RL (2x2), B_RR (2x2)
    # C_L = AL @ B_LL + AR @ B_RL (3x2)
    # C_R = AL @ B_LR + AR @ B_RR (3x2)
    # C = [C_L, C_R]
    
    # We need to decompose the tensor for:
    # C_L = AL @ B_LL + AR @ B_RL
    # C_R = AL @ B_LR + AR @ B_RR
    
    # Each term X @ Y where X is 3x2 and Y is 2x2 has rank 9.
    # Total 4 terms -> 36 rank.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)] # 12 x 36
    V = [[0] * rank for _ in range(m * p)] # 16 x 36
    W = [[0] * rank for _ in range(p * n)] # 12 x 36
    
    col = 0
    
    # Helper to add a 3x2x2 decomposition with rank 9
    def add_3x2x2(A_rows, B_cols, C_rows, offset):
        # A_rows: indices in U for the 3x2 matrix (6 indices)
        # B_cols: indices in V for the 2x2 matrix (4 indices)
        # C_rows: indices in W for the 3x2 result (6 indices)
        # Returns the next column index
        nonlocal col
        
        # Strassen for top 2 rows (2x2x2) -> 7 mults
        # a11, a12; a21, a22
        # b11, b12; b21, b22
        
        # m1 = a11 * b11
        U[A_rows[0]][col] = 1
        V[B_cols[0]][col] = 1
        # c11 += m1, c22 += m1
        W[C_rows[0]][col] = 1
        W[C_rows[5]][col] = 1
        col += 1
        
        # m2 = a11 * (b21 - b11)
        U[A_rows[0]][col] = 1
        V[B_cols[2]][col] = 1
        V[B_cols[0]][col] = -1
        # c21 += m2, c11 -= m2
        W[C_rows[3]][col] = 1
        W[C_rows[0]][col] = -1
        col += 1
        
        # m3 = (a21 + a11) * b11
        U[A_rows[3]][col] = 1
        U[A_rows[0]][col] = 1
        V[B_cols[0]][col] = 1
        # c21 += m3, c22 -= m3
        W[C_rows[3]][col] = 1
        W[C_rows[5]][col] = -1
        col += 1
        
        # m4 = a22 * (b22 - b12)
        U[A_rows[1]][col] = 1
        V[B_cols[3]][col] = 1
        V[B_cols[1]][col] = -1
        # c22 += m4, c11 -= m4
        W[C_rows[5]][col] = 1
        W[C_rows[0]][col] = -1
        col += 1
        
        # m5 = (a11 + a22) * (b11 + b22)
        U[A_rows[0]][col] = 1
        U[A_rows[1]][col] = 1
        V[B_cols[0]][col] = 1
        V[B_cols[3]][col] = 1
        # c11 += m5, c22 += m5
        W[C_rows[0]][col] = 1
        W[C_rows[5]][col] = 1
        col += 1
        
        # m6 = (a12 - a22) * (b11 + b12)
        U[A_rows[1]][col] = 1
        U[A_rows[4]][col] = -1
        V[B_cols[0]][col] = 1
        V[B_cols[1]][col] = 1
        # c22 -= m6, c12 += m6
        W[C_rows[5]][col] = -1
        W[C_rows[1]][col] = 1
        col += 1
        
        # m7 = (a21 - a11) * (b22 - b12)
        U[A_rows[3]][col] = 1
        U[A_rows[0]][col] = -1
        V[B_cols[3]][col] = 1
        V[B_cols[1]][col] = -1
        # c12 += m7, c11 -= m7
        W[C_rows[1]][col] = 1
        W[C_rows[0]][col] = -1
        col += 1
        
        # Row 3:
        # m8 = a31 * b11
        U[A_rows[6]][col] = 1
        V[B_cols[0]][col] = 1
        # c31 += m8
        W[C_rows[6]][col] = 1
        col += 1
        
        # m9 = a32 * b21
        U[A_rows[7]][col] = 1
        V[B_cols[2]][col] = 1
        # c31 += m9
        W[C_rows[6]][col] = 1
        col += 1
        
        # c32 = a31 * b12 + a32 * b22 (2 mults)
        U[A_rows[6]][col] = 1
        V[B_cols[1]][col] = 1
        W[C_rows[7]][col] = 1
        col += 1
        
        U[A_rows[7]][col] = 1
        V[B_cols[3]][col] = 1
        W[C_rows[7]][col] = 1
        col += 1
        
        return col

    # Indices for A (3x4) -> U (12x)
    # A is stored as row-major: i*m + j
    # AL: i=0..2, j=0..1 -> U[0..5]
    # AR: i=0..2, j=2..3 -> U[6..11]
    
    # Indices for B (4x4) -> V (16x)
    # B is stored as row-major: j*p + k
    # B_LL: j=0..1, k=0..1 -> V[0..3]
    # B_LR: j=0..1, k=2..3 -> V[4..7]
    # B_RL: j=2..3, k=0..1 -> V[8..11]
    # B_RR: j=2..3, k=2..3 -> V[12..15]
    
    # Indices for C (3x4) -> W (12x)
    # C is stored as row-major: k*n + i (Trace convention W[k*n+i])
    # Wait, the task says "output factor follows the official trace convention W[k*n+i,r]"
    # This means W is indexed by [k*n + i] where k is col, i is row.
    # C_L is 3x2 (cols 0,1). C_R is 3x2 (cols 2,3).
    # C_L: i=0..2, k=0..1 -> W[k*3+i]
    #   k=0: W[0], W[1], W[2]
    #   k=1: W[3], W[4], W[5]
    # C_R: i=0..2, k=2..3 -> W[k*3+i]
    #   k=2: W[6], W[7], W[8]
    #   k=3: W[9], W[10], W[11]
    
    # Let's map the 3x2 result indices (row-major i*2+k) to W indices (k*3+i)
    # For C_L (k=0,1):
    #   i=0, k=0 -> W[0]
    #   i=0, k=1 -> W[3]
    #   i=1, k=0 -> W[1]
    #   i=1, k=1 -> W[4]
    #   i=2, k=0 -> W[2]
    #   i=2, k=1 -> W[5]
    # My add_3x2x2 assumed C_rows were 0..5 in some order.
    # I need to pass the correct W indices.
    
    # C_L W indices:
    # C[0,0]=W[0], C[0,1]=W[3]
    # C[1,0]=W[1], C[1,1]=W[4]
    # C[2,0]=W[2], C[2,1]=W[5]
    
    # C_R W indices:
    # C[0,2]=W[6], C[0,3]=W[9]
    # C[1,2]=W[7], C[1,3]=W[10]
    # C[2,2]=W[8], C[2,3]=W[11]
    
    # A_L U indices (3x2):
    # A[0,0]=U[0], A[0,1]=U[1]
    # A[1,0]=U[3], A[1,1]=U[4]
    # A[2,0]=U[6], A[2,1]=U[7]
    
    # A_R U indices (3x2):
    # A[0,2]=U[2], A[0,3]=U[5]
    # A[1,2]=U[8], A[1,3]=U[9]
    # A[2,2]=U[10], A[2,3]=U[11]
    
    # B_LL V indices (2x2):
    # B[0,0]=V[0], B[0,1]=V[1]
    # B[1,0]=V[2], B[1,1]=V[3]
    
    # B_LR V indices (2x2):
    # B[0,2]=V[4], B[0,3]=V[5]
    # B[1,2]=V[6], B[1,3]=V[7]
    
    # B_RL V indices (2x2):
    # B[2,0]=V[8], B[2,1]=V[9]
    # B[3,0]=V[10], B[3,1]=V[11]
    
    # B_RR V indices (2x2):
    # B[2,2]=V[12], B[2,3]=V[13]
    # B[3,2]=V[14], B[3,3]=V[15]
    
    # Term 1: AL @ B_LL
    # A_rows: [0, 1, 3, 4, 6, 7] (a11, a12, a21, a22, a31, a32)
    # B_cols: [0, 1, 2, 3] (b11, b12, b21, b22)
    # C_rows: [0, 3, 1, 4, 2, 5] (c11, c12, c21, c22, c31, c32)
    add_3x2x2([0, 1, 3, 4, 6, 7], [0, 1, 2, 3], [0, 3, 1, 4, 2, 5], col)
    
    # Term 2: AR @ B_RL
    # A_rows: [2, 5, 8, 9, 10, 11]
    # B_cols: [8, 9, 10, 11]
    # C_rows: [0, 3, 1, 4, 2, 5] (Same C_L)
    add_3x2x2([2, 5, 8, 9, 10, 11], [8, 9, 10, 11], [0, 3, 1, 4, 2, 5], col)
    
    # Term 3: AL @ B_LR
    # A_rows: [0, 1, 3, 4, 6, 7]
    # B_cols: [4, 5, 6, 7]
    # C_rows: [6, 9, 7, 10, 8, 11] (c11, c12, c21, c22, c31, c32 for C_R)
    add_3x2x2([0, 1, 3, 4, 6, 7], [4, 5, 6, 7], [6, 9, 7, 10, 8, 11], col)
    
    # Term 4: AR @ B_RR
    # A_rows: [2, 5, 8, 9, 10, 11]
    # B_cols: [12, 13, 14, 15]
    # C_rows: [6, 9, 7, 10, 8, 11]
    add_3x2x2([2, 5, 8, 9, 10, 11], [12, 13, 14, 15], [6, 9, 7, 10, 8, 11], col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
