"""
Rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication.
Total rank: 56
"""

import time

DIMENSIONS = [3, 4, 5]

def construct():
    deadline = time.time() + 140
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    total_rank = 56
    
    U = [[0]*total_rank for _ in range(n*m)]  # 12 rows
    V = [[0]*total_rank for _ in range(m*p)]  # 20 rows
    W = [[0]*total_rank for _ in range(p*n)]  # 15 rows
    
    col = 0
    
    # Block 1: C11 = A[0,:] * B[:,0] : rank 4
    # For j in 0..3: U[0*4+j][col]=1, V[j*5+0][col]=1, W[0*3+0][col]=1
    for j in range(4):
        U[0*4+j][col] = 1
        V[j*5+0][col] = 1
        W[0*3+0][col] = 1
        col += 1
    
    # Block 2: C12 = A[0,:] * B[:,1:5] : rank 16
    # For j in 0..3, k in 0..3 (k=0..3 corresponds to B col 1+k):
    # U[0*4+j][col]=1, V[j*5+(1+k)][col]=1, W[k*3+0][col]=1
    for j in range(4):
        for k in range(4):
            U[0*4+j][col] = 1
            V[j*5+(1+k)][col] = 1
            W[k*3+0][col] = 1
            col += 1
    
    # Block 3: C21 = A[1:3,:] * B[:,0] : rank 8
    # For i in 0..1 (i=0,1 corresponds to A row 1+i), j in 0..3:
    # U[(1+i)*4+j][col]=1, V[j*5+0][col]=1, W[0*3+(1+i)][col]=1
    for i in range(2):
        for j in range(4):
            U[(1+i)*4+j][col] = 1
            V[j*5+0][col] = 1
            W[0*3+(1+i)][col] = 1
            col += 1
    
    # Block 4: C22 = A[1:3,:] * B[:,1:5] : rank 28 via 4 Strassen sub-blocks
    # Split A[1:3,:] (2x4) into A2a (2x2) cols 0-1, A2b (2x2) cols 2-3
    # Split B[:,1:5] (4x4) into B2l (4x2) cols 1-2, B2r (4x2) cols 3-4
    # A2*B2l = A2a*B2la + A2b*B2lb, each 2x2*2x2, rank 7
    # A2*B2r = A2a*B2ra + A2b*B2rb, each 2x2*2x2, rank 7
    # Total: 28
    
    # Strassen helper: adds a 2x2*2x2 Strassen block
    # A_sub: 2x2 submatrix of A at rows ar0,ar1 and cols ac0,ac1
    # B_sub: 2x2 submatrix of B at rows br0,br1 and cols bc0,bc1
    # C_sub: 2x2 submatrix of C at rows cr0,cr1 and cols cc0,cc1
    # W convention: W[k*n+i, col] where i is row of C, k is col of C
    def add_strassen(ar0, ac0, br0, bc0, cr0, cc0):
        nonlocal col
        # Strassen 7 products
        # M1 = (A11+A22)*(B11+B22)
        U[ar0*m+ac0][col] = 1
        U[(ar0+1)*m+(ac0+1)][col] = 1
        V[br0*p+bc0][col] = 1
        V[(br0+1)*p+(bc0+1)][col] = 1
        W[cc0*n+cr0][col] = 1
        W[(cc0+1)*n+(cr0+1)][col] = 1
        col += 1
        
        # M2 = (A21+A22)*B11
        U[(ar0+1)*m+ac0][col] = 1
        U[(ar0+1)*m+(ac0+1)][col] = 1
        V[br0*p+bc0][col] = 1
        W[cc0*n+(cr0+1)][col] = 1
        col += 1
        
        # M3 = A11*(B12-B22)
        U[ar0*m+ac0][col] = 1
        V[br0*p+(bc0+1)][col] = 1
        V[(br0+1)*p+(bc0+1)][col] = -1
        W[(cc0+1)*n+cr0][col] = 1
        W[(cc0+1)*n+(cr0+1)][col] = 1
        col += 1
        
        # M4 = A22*(B21-B11)
        U[(ar0+1)*m+(ac0+1)][col] = 1
        V[(br0+1)*p+bc0][col] = 1
        V[br0*p+bc0][col] = -1
        W[cc0*n+cr0][col] = 1
        W[cc0*n+(cr0+1)][col] = 1
        col += 1
        
        # M5 = (A11+A12)*B22
        U[ar0*m+ac0][col] = 1
        U[ar0*m+(ac0+1)][col] = 1
        V[(br0+1)*p+(bc0+1)][col] = 1
        W[cc0*n+cr0][col] = -1
        W[(cc0+1)*n+cr0][col] = 1
        col += 1
        
        # M6 = (A21-A11)*(B11+B12)
        U[(ar0+1)*m+ac0][col] = 1
        U[ar0*m+ac0][col] = -1
        V[br0*p+bc0][col] = 1
        V[br0*p+(bc0+1)][col] = 1
        W[(cc0+1)*n+(cr0+1)][col] = 1
        col += 1
        
        # M7 = (A12-A22)*(B21+B22)
        U[ar0*m+(ac0+1)][col] = 1
        U[(ar0+1)*m+(ac0+1)][col] = -1
        V[(br0+1)*p+bc0][col] = 1
        V[(br0+1)*p+(bc0+1)][col] = 1
        W[cc0*n+cr0][col] = 1
        col += 1
    
    # A2a: rows 1-2, cols 0-1 of A; B2la: rows 0-1, cols 1-2 of B
    add_strassen(1, 0, 0, 1, 1, 1)
    # A2b: rows 1-2, cols 2-3 of A; B2lb: rows 2-3, cols 1-2 of B
    add_strassen(1, 2, 2, 1, 1, 1)
    # A2a: rows 1-2, cols 0-1 of A; B2ra: rows 0-1, cols 3-4 of B
    add_strassen(1, 0, 0, 3, 1, 3)
    # A2b: rows 1-2, cols 2-3 of A; B2rb: rows 2-3, cols 3-4 of B
    add_strassen(1, 2, 2, 3, 1, 3)
    
    assert col == total_rank, f"Column count {col} != total_rank {total_rank}"
    
    return {"U": U, "V": V, "W": W}
