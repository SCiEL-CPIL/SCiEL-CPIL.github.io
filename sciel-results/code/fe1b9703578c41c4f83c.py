import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # We decompose M_{4,5,6} into blocks.
    # m=5 -> 2 + 3
    # n=4 -> 2 + 2
    # p=6 -> 2 + 2 + 2
    
    # Rank components:
    # 6 blocks of M_{2,2,2} (Rank 7 each) -> 42
    # 6 blocks of M_{2,3,2} (Rank 11 each) -> 66
    # Total Rank = 108
    
    rank = 108
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen_block(u_offset, v_offset, w_offset, rank_offset):
        # u_offset: (i, j) start indices for A (2x2)
        # v_offset: (j, k) start indices for B (2x2)
        # w_offset: (k, i) start indices for C (2x2)
        # rank_offset: start column index in U,V,W
        
        i0, j0 = u_offset
        j0v, k0 = v_offset
        k0w, i0w = w_offset
        
        # Strassen terms
        # M1 = (A11+A22)(B11+B22)
        U[i0*m + j0, rank_offset] = 1
        U[(i0+1)*m + (j0+1), rank_offset] = 1
        V[j0*p + k0, rank_offset] = 1
        V[(j0+1)*p + (k0+1), rank_offset] = 1
        W[k0*n + i0, rank_offset] = 1
        W[(k0+1)*n + (i0+1), rank_offset] = 1
        W[k0*n + (i0+1), rank_offset] = 1
        W[(k0+1)*n + i0, rank_offset] = 1
        # Note: W indices for M1:
        # C11 += 1, C12 += 0, C21 += 0, C22 += 1 ? 
        # Strassen:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M6
        
        # Let's map W coefficients for each M term.
        # M1: C11+1, C22+1
        # M2: C21+1, C22-1
        # M3: C12+1
        # M4: C11+1, C21+1
        # M5: C11-1, C12+1
        # M6: C22+1
        # M7: C11+1
        
        # M1
        W[k0*n + i0, rank_offset] = 1
        W[(k0+1)*n + (i0+1), rank_offset] = 1
        col_local = rank_offset + 1
        
        # M2 = (A11+A12)B22
        U[i0*m + j0, col_local] = 1
        U[i0*m + (j0+1), col_local] = 1
        V[(j0+1)*p + (k0+1), col_local] = 1
        W[(k0+1)*n + i0, col_local] = -1 # C22 -1
        W[(k0+1)*n + (i0+1), col_local] = 1 # C21 +1 (Wait, C21 is row 2 col 1. Index (k0+1, i0)? No. C is p x n. C[k,i]. C21 is k=1, i=0? No. C is 2x2. C[1,0] is C21. C[0,1] is C12.)
        # C indices: C[0,0]=C11, C[0,1]=C12, C[1,0]=C21, C[1,1]=C22.
        # W[k*n+i].
        # C21 is k=1, i=0. W[1*n+0].
        # C22 is k=1, i=1. W[1*n+1].
        W[(k0+1)*n + i0, col_local] = 1 # C21
        W[(k0+1)*n + (i0+1), col_local] = -1 # C22
        col_local += 1
        
        # M3 = A11(B12-B22)
        U[i0*m + j0, col_local] = 1
        V[j0*p + (k0+1), col_local] = 1
        V[(j0+1)*p + (k0+1), col_local] = -1
        W[k0*n + (i0+1), col_local] = 1 # C12
        col_local += 1
        
        # M4 = A22(B21-B11)
        U[(i0+1)*m + (j0+1), col_local] = 1
        V[(j0+1)*p + k0, col_local] = 1
        V[(j0+1)*p + (k0+1), col_local] = -1
        W[k0*n + i0, col_local] = 1 # C11
        W[(k0+1)*n + i0, col_local] = 1 # C21
        col_local += 1
        
        # M5 = (A21+A11)B11
        U[(i0+1)*m + j0, col_local] = 1
        U[i0*m + j0, col_local] = 1
        V[j0*p + k0, col_local] = 1
        W[k0*n + i0, col_local] = -1 # C11
        W[k0*n + (i0+1), col_local] = 1 # C12
        col_local += 1
        
        # M6 = (A22-A11)(B12+B22)
        U[(i0+1)*m + (j0+1), col_local] = 1
        U[i0*m + j0, col_local] = -1
        V[j0*p + (k0+1), col_local] = 1
        V[(j0+1)*p + (k0+1), col_local] = 1
        W[(k0+1)*n + (i0+1), col_local] = 1 # C22
        col_local += 1
        
        # M7 = (A11-A21)(B21+B11)
        U[i0*m + j0, col_local] = 1
        U[(i0+1)*m + j0, col_local] = -1
        V[j0*p + k0, col_local] = 1
        V[(j0+1)*p + k0, col_local] = 1
        W[k0*n + i0, col_local] = 1 # C11
        col_local += 1
        
        return col_local

    # Helper to add a 2x3x2 block (Rank 11)
    def add_232_block(u_offset, v_offset, w_offset, rank_offset):
        # u_offset: (i, j) start for A (2x3)
        # v_offset: (j, k) start for B (3x2)
        # w_offset: (k, i) start for C (2x2)
        i0, j0 = u_offset
        j0v, k0 = v_offset
        k0w, i0w = w_offset
        
        # Part 1: 2x1x2 (Rank 4) using columns j0, j0+1? No.
        # Split B's 3 rows into 1 and 2.
        # B row j0 is 1x2. B rows j0+1, j0+2 is 2x2.
        # A col j0 is 2x1. A cols j0+1, j0+2 is 2x2.
        
        # Term 1: A_col0 * B_row0 (2x1 * 1x2)
        # C11 += A11*B11
        U[i0*m + j0, rank_offset] = 1
        V[j0*p + k0, rank_offset] = 1
        W[k0*n + i0, rank_offset] = 1
        col_local = rank_offset + 1
        
        # C12 += A11*B12
        U[i0*m + j0, col_local] = 1
        V[j0*p + (k0+1), col_local] = 1
        W[(k0+1)*n + i0, col_local] = 1
        col_local += 1
        
        # C21 += A21*B11
        U[(i0+1)*m + j0, col_local] = 1
        V[j0*p + k0, col_local] = 1
        W[k0*n + (i0+1), col_local] = 1
        col_local += 1
        
        # C22 += A21*B12
        U[(i0+1)*m + j0, col_local] = 1
        V[j0*p + (k0+1), col_local] = 1
        W[(k0+1)*n + (i0+1), col_local] = 1
        col_local += 1
        
        # Part 2: A_2x2 * B_2x2 (Strassen)
        # A uses cols j0+1, j0+2. B uses rows j0+1, j0+2.
        # U offset i0, j0+1. V offset j0+1, k0. W offset k0, i0.
        col_local = add_strassen_block((i0, j0+1), (j0+1, k0), (k0, i0), col_local)
        
        return col_local

    # Block 1: A1 (4x2) * B1 (2x6)
    # Split A1 into A11 (2x2) and A21 (2x2)
    # Split B1 into B11 (2x2), B12 (2x2), B13 (2x2)
    
    # A11: i=0,1; j=0,1
    # B11: j=0,1; k=0,1
    col = add_strassen_block((0,0), (0,0), (0,0), col)
    # B12: j=0,1; k=2,3
    col = add_strassen_block((0,0), (0,2), (2,0), col)
    # B13: j=0,1; k=4,5
    col = add_strassen_block((0,0), (0,4), (4,0), col)
    
    # A21: i=2,3; j=0,1
    # B11: j=0,1; k=0,1
    col = add_strassen_block((2,0), (0,0), (0,2), col)
    # B12: j=0,1; k=2,3
    col = add_strassen_block((2,0), (0,2), (2,2), col)
    # B13: j=0,1; k=4,5
    col = add_strassen_block((2,0), (0,4), (4,2), col)
    
    # Block 2: A2 (4x3) * B2 (3x6)
    # Split A2 into A12 (2x3) and A22 (2x3)
    # Split B2 into B21 (3x2), B22 (3x2), B23 (3x2)
    
    # A12: i=0,1; j=2,3,4
    # B21: j=2,3,4; k=0,1
    col = add_232_block((0,2), (2,0), (0,0), col)
    # B22: j=2,3,4; k=2,3
    col = add_232_block((0,2), (2,2), (2,0), col)
    # B23: j=2,3,4; k=4,5
    col = add_232_block((0,2), (2,4), (4,0), col)
    
    # A22: i=2,3; j=2,3,4
    # B21: j=2,3,4; k=0,1
    col = add_232_block((2,2), (2,0), (0,2), col)
    # B22: j=2,3,4; k=2,3
    col = add_232_block((2,2), (2,2), (2,2), col)
    # B23: j=2,3,4; k=4,5
    col = add_232_block((2,2), (2,4), (4,2), col)
    
    return {"U": U, "V": V, "W": W}
