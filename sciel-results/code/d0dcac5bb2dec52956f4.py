import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # We will construct a decomposition of rank 50.
    # Strategy:
    # Split A (2x4) into A1 (2x2) and A2 (2x2).
    # Split B (4x7) into B1 (2x7) and B2 (2x7).
    # C = A1*B1 + A2*B2.
    # Compute A1*B1 and A2*B2 separately.
    # Each is 2x2x7.
    # For 2x2x7, split the 7 columns into 2+2+2+1.
    # Use Strassen (rank 7) for the 2x2x2 blocks.
    # Use Schoolbook (rank 4) for the 2x2x1 block.
    # Rank for one 2x2x7 is 3*7 + 4 = 25.
    # Total rank = 25 + 25 = 50.
    
    rank = 50
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    
    # Helper to add a Strassen block for 2x2x2
    # Indices for A block: i in {0,1}, j in {j_off, j_off+1}
    # Indices for B block: j in {j_off, j_off+1}, k in {k_off, k_off+1}
    # Indices for C block: i in {0,1}, k in {k_off, k_off+1}
    def add_strassen_block(i_off, j_off, k_off, col_start):
        # Strassen 7 terms for 2x2x2
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B11)
        # M4 = A11(B01 - B00)
        # M5 = (A00 + A10)B11
        # M6 = (A11 - A10)(B00 + B10)
        # M7 = (A01 - A11)(B01 + B11)
        
        # M1
        U[i_off*4 + j_off][col_start] = 1
        U[i_off*4 + j_off+1][col_start] = 1
        U[(i_off+1)*4 + j_off][col_start] = 1
        U[(i_off+1)*4 + j_off+1][col_start] = 1
        
        V[j_off*7 + k_off][col_start] = 1
        V[j_off*7 + k_off+1][col_start] = 1
        V[(j_off+1)*7 + k_off][col_start] = 1
        V[(j_off+1)*7 + k_off+1][col_start] = 1
        
        W[k_off*2 + i_off][col_start] = 1
        W[k_off*2 + i_off+1][col_start] = 1
        W[(k_off+1)*2 + i_off][col_start] = 1
        W[(k_off+1)*2 + i_off+1][col_start] = 1
        col_start += 1
        
        # M2
        U[i_off*4 + j_off][col_start] = 1
        U[i_off*4 + j_off+1][col_start] = 1
        U[(i_off+1)*4 + j_off][col_start] = 1
        U[(i_off+1)*4 + j_off+1][col_start] = 1
        
        V[j_off*7 + k_off][col_start] = 1
        V[j_off*7 + k_off+1][col_start] = 1
        
        W[k_off*2 + i_off][col_start] = 1
        W[k_off*2 + i_off+1][col_start] = 1
        W[(k_off+1)*2 + i_off][col_start] = 1
        W[(k_off+1)*2 + i_off+1][col_start] = 1
        col_start += 1
        
        # M3
        U[i_off*4 + j_off][col_start] = 1
        U[i_off*4 + j_off+1][col_start] = 1
        
        V[j_off*7 + k_off][col_start] = 1
        V[j_off*7 + k_off+1][col_start] = -1
        
        W[k_off*2 + i_off][col_start] = 1
        W[k_off*2 + i_off+1][col_start] = 1
        W[(k_off+1)*2 + i_off][col_start] = 1
        W[(k_off+1)*2 + i_off+1][col_start] = 1
        col_start += 1
        
        # M4
        U[i_off*4 + j_off][col_start] = 1
        U[i_off*4 + j_off+1][col_start] = 1
        
        V[j_off*7 + k_off][col_start] = 1
        V[j_off*7 + k_off+1][col_start] = -1
        
        W[k_off*2 + i_off][col_start] = 1
        W[k_off*2 + i_off+1][col_start] = 1
        W[(k_off+1)*2 + i_off][col_start] = 1
        W[(k_off+1)*2 + i_off+1][col_start] = 1
        col_start += 1
        
        # M5
        U[i_off*4 + j_off][col_start] = 1
        U[i_off*4 + j_off+1][col_start] = 1
        U[(i_off+1)*4 + j_off][col_start] = 1
        U[(i_off+1)*4 + j_off+1][col_start] = 1
        
        V[j_off*7 + k_off][col_start] = 1
        V[j_off*7 + k_off+1][col_start] = 1
        
        W[k_off*2 + i_off][col_start] = 1
        W[k_off*2 + i_off+1][col_start] = 1
        W[(k_off+1)*2 + i_off][col_start] = 1
        W[(k_off+1)*2 + i_off+1][col_start] = 1
        col_start += 1
        
        # M6
        U[i_off*4 + j_off][col_start] = 1
        U[i_off*4 + j_off+1][col_start] = 1
        U[(i_off+1)*4 + j_off][col_start] = -1
        U[(i_off+1)*4 + j_off+1][col_start] = 1
        
        V[j_off*7 + k_off][col_start] = 1
        V[j_off*7 + k_off+1][col_start] = 1
        
        W[k_off*2 + i_off][col_start] = 1
        W[k_off*2 + i_off+1][col_start] = 1
        W[(k_off+1)*2 + i_off][col_start] = 1
        W[(k_off+1)*2 + i_off+1][col_start] = 1
        col_start += 1
        
        # M7
        U[i_off*4 + j_off][col_start] = 1
        U[i_off*4 + j_off+1][col_start] = 1
        U[(i_off+1)*4 + j_off][col_start] = 1
        U[(i_off+1)*4 + j_off+1][col_start] = -1
        
        V[j_off*7 + k_off][col_start] = 1
        V[j_off*7 + k_off+1][col_start] = 1
        
        W[k_off*2 + i_off][col_start] = 1
        W[k_off*2 + i_off+1][col_start] = 1
        W[(k_off+1)*2 + i_off][col_start] = 1
        W[(k_off+1)*2 + i_off+1][col_start] = 1
        col_start += 1
        
        return col_start

    # Helper to add a Schoolbook block for 2x2x1
    def add_schoolbook_block(i_off, j_off, k_off, col_start):
        # 4 terms
        for di in range(2):
            for dj in range(2):
                U[(i_off+di)*4 + (j_off+dj)][col_start] = 1
                V[(j_off+dj)*7 + k_off][col_start] = 1
                W[k_off*2 + (i_off+di)][col_start] = 1
                col_start += 1
        return col_start

    # Process A1 (i=0,1; j=0,1) and B1 (j=0,1; k=0..6)
    # B1 columns: 0,1 (Strassen), 2,3 (Strassen), 4,5 (Strassen), 6 (Schoolbook)
    col = add_strassen_block(0, 0, 0, col)
    col = add_strassen_block(0, 0, 2, col)
    col = add_strassen_block(0, 0, 4, col)
    col = add_schoolbook_block(0, 0, 6, col)
    
    # Process A2 (i=0,1; j=2,3) and B2 (j=2,3; k=0..6)
    # B2 columns: 0,1 (Strassen), 2,3 (Strassen), 4,5 (Strassen), 6 (Schoolbook)
    col = add_strassen_block(0, 2, 0, col)
    col = add_strassen_block(0, 2, 2, col)
    col = add_strassen_block(0, 2, 4, col)
    col = add_schoolbook_block(0, 2, 6, col)
    
    return {"U": U, "V": V, "W": W}
