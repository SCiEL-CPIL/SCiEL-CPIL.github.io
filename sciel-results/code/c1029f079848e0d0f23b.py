"""
Rank-22 exact decomposition for 2x3 * 3x4 matrix multiplication.

Construction:
  Split A (2x3) into A_L (2x2, cols 0,1) and A_R (2x1, col 2).
  Split B (3x4) into B_TL (2x2, rows 0,1; cols 0,1), B_TR (2x2, rows 0,1; cols 2,3),
                    B_BL (1x2, row 2; cols 0,1), B_BR (1x2, row 2; cols 2,3).

  C = A_L * B_TL + A_L * B_TR + A_R * B_BL + A_R * B_BR

  Use Strassen (rank 7) for each 2x2 * 2x2 block.
  Use schoolbook (rank 4) for each 2x1 * 1x2 block.

  Total rank = 7 + 7 + 4 + 4 = 22.
"""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START

# Strassen rank-7 factors for 2x2 * 2x2 multiplication.
# A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = A*B
# Standard Strassen:
#   M1 = (a+d)(e+h)
#   M2 = (c+d)e
#   M3 = a(f-h)
#   M4 = d(g-e)
#   M5 = (a+b)h
#   M6 = (c-a)(e+f)
#   M7 = (b-d)(g+h)
#   c11 = M1 - M4 + M5 + M6
#   c12 = M3 + M4
#   c21 = M2 + M5
#   c22 = M1 - M2 + M3 + M7

def _strassen_factors():
    """
    Return (U, V, W) for 2x2 * 2x2 with Strassen rank 7.
    U: shape (4, 7) for A (row-major: a, b, c, d)
    V: shape (4, 7) for B (row-major: e, f, g, h)
    W: shape (4, 7) for C (row-major: c11, c12, c21, c22)
    """
    # A indices: 0=a, 1=b, 2=c, 3=d
    # B indices: 0=e, 1=f, 2=g, 3=h
    # C indices: 0=c11, 1=c12, 2=c21, 3=c22
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a+d)(e+h): U[a]=1, U[d]=1, V[e]=1, V[h]=1, W[c11]=1, W[c22]=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (c+d)e: U[c]=1, U[d]=1, V[e]=1, W[c21]=1, W[c22]=-1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = a(f-h): U[a]=1, V[f]=1, V[h]=-1, W[c12]=1, W[c22]=1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = d(g-e): U[d]=1, V[g]=1, V[e]=-1, W[c11]=-1, W[c12]=1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[1][3] = 1
    
    # M5 = (a+b)h: U[a]=1, U[b]=1, V[h]=1, W[c11]=1, W[c21]=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[2][4] = 1
    
    # M6 = (c-a)(e+f): U[c]=1, U[a]=-1, V[e]=1, V[f]=1, W[c11]=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = 1
    
    # M7 = (b-d)(g+h): U[b]=1, U[d]=-1, V[g]=1, V[h]=1, W[c22]=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[3][6] = 1
    
    return U, V, W


def _schoolbook_2x1_x_1x2_factors():
    """
    Rank-4 schoolbook for (2x1) * (1x2) = 2x2.
    A = [[x1], [x2]] (2x1), B = [[y1, y2]] (1x2)
    C = [[x1*y1, x1*y2], [x2*y1, x2*y2]]
    
    U: shape (2, 4) for A (indices: 0=x1, 1=x2)
    V: shape (2, 4) for B (indices: 0=y1, 1=y2)
    W: shape (4, 4) for C (indices: 0=c11, 1=c12, 2=c21, 3=c22)
    """
    U = [[0]*4 for _ in range(2)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(4)]
    
    # c11 = x1*y1
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    # c12 = x1*y2
    U[0][1] = 1; V[1][1] = 1; W[1][1] = 1
    # c21 = x2*y1
    U[1][2] = 1; V[0][2] = 1; W[2][2] = 1
    # c22 = x2*y2
    U[1][3] = 1; V[1][3] = 1; W[3][3] = 1
    
    return U, V, W


def _embed_block_u(u_block, row_offset):
    """Embed a 4x7 or 2x4 block U into the full 6x22 U matrix at given row offset."""
    # Full U has shape (6, 22): rows correspond to A entries (2x3 = 6)
    # u_block has shape (num_rows, num_cols)
    num_rows, num_cols = len(u_block), len(u_block[0])
    return u_block, row_offset, num_rows, num_cols


def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # Full U: (n*m, rank) = (6, 22)
    # Full V: (m*p, rank) = (12, 22)
    # Full W: (p*n, rank) = (8, 22)
    rank = 22
    U = [[0] * rank for _ in range(n * m)]   # 6 rows
    V = [[0] * rank for _ in range(m * p)]   # 12 rows
    W = [[0] * rank for _ in range(p * n)]   # 8 rows
    
    # Get base factor blocks
    sU, sV, sW = _strassen_factors()       # each 4x7
    bU, bV, bW = _schoolbook_2x1_x_1x2_factors()  # bU: 2x4, bV: 2x4, bW: 4x4
    
    # Column offsets for the 4 sub-problems:
    #   0-6:   A_L * B_TL (2x2 * 2x2)
    #   7-13:  A_L * B_TR (2x2 * 2x2)
    #   14-17: A_R * B_BL (2x1 * 1x2)
    #   18-21: A_R * B_BR (2x1 * 1x2)
    
    col_offsets = [0, 7, 14, 18]
    
    # --- Block 1: A_L (rows 0,1 of A) * B_TL (rows 0,1 of B; cols 0,1) ---
    # A_L rows: 0,1 -> U rows 0,1
    # B_TL: B rows 0,1; B cols 0,1 -> V rows 0,1 (B row 0 cols 0,1 are V rows 0,1)
    #        B row 1 cols 0,1 are V rows 3,4
    # C block: C rows 0,1; C cols 0,1 -> W: W[k*n+i, col] where k is C row, i is C col
    #   C[0,0] -> W[0*2+0]=W[0], C[0,1] -> W[1], C[1,0] -> W[2], C[1,1] -> W[3]
    
    for c in range(7):
        col = col_offsets[0] + c
        # U: A_L entries (a,b,c,d) map to A rows 0,1
        #   a=A[0,0]->U[0], b=A[0,1]->U[1], c=A[1,0]->U[2], d=A[1,1]->U[3]
        U[0][col] += sU[0][c]
        U[1][col] += sU[1][c]
        U[2][col] += sU[2][c]
        U[3][col] += sU[3][c]
        
        # V: B_TL entries (e,f,g,h) where B_TL = [[e,f],[g,h]] = B[0:2, 0:2]
        #   e=B[0,0]->V[0], f=B[0,1]->V[1], g=B[1,0]->V[3], h=B[1,1]->V[4]
        V[0][col] += sV[0][c]
        V[1][col] += sV[1][c]
        V[3][col] += sV[2][c]
        V[4][col] += sV[3][c]
        
        # W: C block rows 0,1 cols 0,1
        #   c11->W[0], c12->W[1], c21->W[2], c22->W[3]
        W[0][col] += sW[0][c]
        W[1][col] += sW[1][c]
        W[2][col] += sW[2][c]
        W[3][col] += sW[3][c]
    
    # --- Block 2: A_L (rows 0,1) * B_TR (rows 0,1; cols 2,3) ---
    for c in range(7):
        col = col_offsets[1] + c
        U[0][col] += sU[0][c]
        U[1][col] += sU[1][c]
        U[2][col] += sU[2][c]
        U[3][col] += sU[3][c]
        
        # B_TR = B[0:2, 2:4] = [[e,f],[g,h]]
        #   e=B[0,2]->V[2], f=B[0,3]->V[3], g=B[1,2]->V[5], h=B[1,3]->V[6]
        V[2][col] += sV[0][c]
        V[3][col] += sV[1][c]
        V[5][col] += sV[2][c]
        V[6][col] += sV[3][c]
        
        # C block: rows 0,1; cols 2,3
        #   C[0,2]->W[4], C[0,3]->W[5], C[1,2]->W[6], C[1,3]->W[7]
        W[4][col] += sW[0][c]
        W[5][col] += sW[1][c]
        W[6][col] += sW[2][c]
        W[7][col] += sW[3][c]
    
    # --- Block 3: A_R (row 2, i.e. A[*,2]) * B_BL (row 2; cols 0,1) ---
    # A_R is 2x1: entries A[0,2]=x1, A[1,2]=x2 -> U rows 4,5
    # B_BL is 1x2: B[2,0]=y1, B[2,1]=y2 -> V rows 6,7
    # C block: rows 0,1; cols 0,1 -> W[0..3]
    
    for c in range(4):
        col = col_offsets[2] + c
        # bU: 2x4, rows are x1, x2
        U[4][col] += bU[0][c]
        U[5][col] += bU[1][c]
        
        # bV: 2x4, rows are y1, y2
        V[6][col] += bV[0][c]
        V[7][col] += bV[1][c]
        
        # bW: 4x4, rows are c11, c12, c21, c22
        W[0][col] += bW[0][c]
        W[1][col] += bW[1][c]
        W[2][col] += bW[2][c]
        W[3][col] += bW[3][c]
    
    # --- Block 4: A_R (row 2) * B_BR (row 2; cols 2,3) ---
    for c in range(4):
        col = col_offsets[3] + c
        U[4][col] += bU[0][c]
        U[5][col] += bU[1][c]
        
        # B_BR = B[2, 2:4] = [[y1, y2]] where y1=B[2,2], y2=B[2,3]
        # V rows: B[2,2]->V[8], B[2,3]->V[9]
        V[8][col] += bV[0][c]
        V[9][col] += bV[1][c]
        
        # C block: rows 0,1; cols 2,3 -> W[4..7]
        W[4][col] += bW[0][c]
        W[5][col] += bW[1][c]
        W[6][col] += bW[2][c]
        W[7][col] += bW[3][c]
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
