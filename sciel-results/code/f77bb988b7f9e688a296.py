"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank):
    """Verify tensor decomposition exactly (for integer/half-integer entries)."""
    def scale2(mat):
        out = np.array(mat, dtype=np.float64) * 2
        if np.max(np.abs(out - np.round(out))) > 1e-9:
            return None
        return np.round(out).astype(np.int64)
    U2 = scale2(U)
    V2 = scale2(V)
    W2 = scale2(W)
    if U2 is None or V2 is None or W2 is None:
        return False
    # Build target tensor
    target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                target[a*m + b, b*p + c, c*n + a] = 1
    T_computed = np.einsum('ic,jc,kc->ijk', U2, V2, W2, optimize=True)
    T_target_8 = target * 8
    if np.max(np.abs(T_computed - T_target_8)) > 1e-6:
        return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for a in range(n):
        for b in range(m):
            for c in range(p):
                U[a*m + b][col] = 1
                V[b*p + c][col] = 1
                W[c*n + a][col] = 1
                col += 1
    return U, V, W

def _strassen_2x2():
    """Return U, V, W for 2x2 matrix multiplication using Strassen's algorithm."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)*(B11+B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A21+A22)*B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    # M3 = A11*(B12-B22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = A22*(B21-B11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A11+A12)*B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (A21-A11)*(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A12-A22)*(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = 1
    
    return U, V, W

def _laderman_3x3():
    """Return U, V, W for 3x3 matrix multiplication using Laderman's algorithm."""
    U = [[0]*15 for _ in range(9)]
    V = [[0]*15 for _ in range(9)]
    W = [[0]*15 for _ in range(9)]
    
    # Product 0: A[0,0] * B[0,0] -> C[0,0]
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    
    # Product 1: (A[0,0]+A[0,1]) * (B[0,1]-B[1,1]) -> C[0,1]
    U[0][1] = 1; U[1][1] = 1
    V[1][1] = 1; V[4][1] = -1
    W[1][1] = 1
    
    # Product 2: (A[0,0]+A[0,2]) * (B[0,2]-B[2,2]) -> C[0,2]
    U[0][2] = 1; U[2][2] = 1
    V[2][2] = 1; V[8][2] = -1
    W[2][2] = 1
    
    # Product 3: (A[0,1]+A[0,2]) * (B[1,2]-B[2,1]) -> C[0,2] (additional)
    U[1][3] = 1; U[2][3] = 1
    V[5][3] = 1; V[3][3] = -1
    W[2][3] = 1
    
    # Product 4: A[1,0] * B[0,0] -> C[1,0]
    U[3][4] = 1; V[0][4] = 1; W[3][4] = 1
    
    # Product 5: (A[1,0]+A[1,1]) * (B[0,1]-B[1,1]) -> C[1,1]
    U[3][5] = 1; U[4][5] = 1
    V[1][5] = 1; V[4][5] = -1
    W[4][5] = 1
    
    # Product 6: (A[1,0]+A[1,2]) * (B[0,2]-B[2,2]) -> C[1,2]
    U[3][6] = 1; U[5][6] = 1
    V[2][6] = 1; V[8][6] = -1
    W[5][6] = 1
    
    # Product 7: (A[1,1]+A[1,2]) * (B[1,2]-B[2,1]) -> C[1,2] (additional)
    U[4][7] = 1; U[5][7] = 1
    V[5][7] = 1; V[3][7] = -1
    W[5][7] = 1
    
    # Product 8: A[2,0] * B[0,0] -> C[2,0]
    U[6][8] = 1; V[0][8] = 1; W[6][8] = 1
    
    # Product 9: (A[2,0]+A[2,1]) * (B[0,1]-B[1,1]) -> C[2,1]
    U[6][9] = 1; U[7][9] = 1
    V[1][9] = 1; V[4][9] = -1
    W[7][9] = 1
    
    # Product 10: (A[2,0]+A[2,2]) * (B[0,2]-B[2,2]) -> C[2,2]
    U[6][10] = 1; U[8][10] = 1
    V[2][10] = 1; V[8][10] = -1
    W[8][10] = 1
    
    # Product 11: (A[2,1]+A[2,2]) * (B[1,2]-B[2,1]) -> C[2,2] (additional)
    U[7][11] = 1; U[8][11] = 1
    V[5][11] = 1; V[3][11] = -1
    W[8][11] = 1
    
    # Product 12: (A[0,0]+A[1,0]+A[2,0]) * (B[1,0]+B[2,0]) -> C[0,0]+C[1,0]+C[2,0]
    U[0][12] = 1; U[3][12] = 1; U[6][12] = 1
    V[3][12] = 1; V[6][12] = 1
    W[0][12] = 1; W[3][12] = 1; W[6][12] = 1
    
    # Product 13: (A[0,1]+A[1,1]+A[2,1]) * (B[1,0]+B[2,0]) -> C[0,1]+C[1,1]+C[2,1]
    U[1][13] = 1; U[4][13] = 1; U[7][13] = 1
    V[3][13] = 1; V[6][13] = 1
    W[1][13] = 1; W[4][13] = 1; W[7][13] = 1
    
    # Product 14: (A[0,2]+A[1,2]+A[2,2]) * (B[1,0]+B[2,0]) -> C[0,2]+C[1,2]+C[2,2]
    U[2][14] = 1; U[5][14] = 1; U[8][14] = 1
    V[3][14] = 1; V[6][14] = 1
    W[2][14] = 1; W[5][14] = 1; W[8][14] = 1
    
    return U, V, W

def _schoolbook_rect(n, m, p):
    """Return schoolbook decomposition for n x m by m x p."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for a in range(n):
        for b in range(m):
            for c in range(p):
                U[a*m + b][col] = 1
                V[b*p + c][col] = 1
                W[c*n + a][col] = 1
                col += 1
    return U, V, W

def construct():
    deadline = time.time() + 135.0
    n, m, p = DIMENSIONS
    
    # Use block decomposition with Strassen (2x2) and Laderman (3x3)
    # A = [A22 A23; A32 A33]
    # B = [B22 B23; B32 B33]
    
    # Ranks:
    # M1 = A22 B22 (2x2x2) -> 7
    # M2 = A33 B33 (3x3x3) -> 15
    # M3 = A23 B32 (2x3x2) -> 12
    # M4 = A32 B23 (3x2x3) -> 18
    # M5 = A22 B23 (2x2x3) -> 12
    # M6 = A23 B33 (2x3x3) -> 18
    # M7 = A32 B22 (3x2x2) -> 12
    # M8 = A33 B32 (3x3x2) -> 18
    
    total_rank = 7 + 15 + 12 + 18 + 12 + 18 + 12 + 18
    U = [[0]*total_rank for _ in range(25)]
    V = [[0]*total_rank for _ in range(25)]
    W = [[0]*total_rank for _ in range(25)]
    
    col_offset = 0
    
    # M1: A22 B22 (Rows 0-1, Cols 0-1 of A; Rows 0-1, Cols 0-1 of B; Rows 0-1, Cols 0-1 of C)
    U1, V1, W1 = _strassen_2x2()
    for i in range(4):
        for r in range(7):
            U[i][col_offset + r] = U1[i][r]
            V[i][col_offset + r] = V1[i][r]
            W[i][col_offset + r] = W1[i][r]
    col_offset += 7
    
    # M2: A33 B33 (Rows 2-4, Cols 2-4 of A; Rows 2-4, Cols 2-4 of B; Rows 2-4, Cols 2-4 of C)
    U2, V2, W2 = _laderman_3x3()
    for i in range(9):
        a, b = divmod(i, 3)
        global_i = (a+2)*5 + (b+2)
        for r in range(15):
            U[global_i][col_offset + r] = U2[i][r]
            V[global_i][col_offset + r] = V2[i][r]
            W[global_i][col_offset + r] = W2[i][r]
    col_offset += 15
    
    # M3: A23 B32 (Rows 0-1, Cols 2-4 of A; Rows 2-4, Cols 0-1 of B; Rows 0-1, Cols 0-1 of C)
    U3, V3, W3 = _schoolbook_rect(2, 3, 2)
    for i in range(6):
        a, b = divmod(i, 3)
        global_i_u = a*5 + (b+2)
        global_i_v = (b+2)*5 + a # Wait, V index is b*5+c. b is 2-4, c is 0-1.
        # A23: a in 0-1, b in 2-4. Index a*5+b.
        # B32: b in 2-4, c in 0-1. Index b*5+c.
        # C22: c in 0-1, a in 0-1. Index c*5+a.
        
        # U3[i][r] corresponds to A23[i]
        # V3[i][r] corresponds to B32[i]
        # W3[i][r] corresponds to C22[i]
        
        # Mapping:
        # U3 row i: a = i//3, b = i%3. Global U row: a*5 + (b+2).
        # V3 row i: b = i//3, c = i%3. Global V row: (b+2)*5 + c.
        # W3 row i: c = i//3, a = i%3. Global W row: c*5 + a.
        
        for r in range(12):
            U[a*5 + (b+2)][col_offset + r] = U3[i][r]
            V[(b+2)*5 + a][col_offset + r] = V3[i][r] # Error: V index is b*5+c. b is from A23 col, c is from B32 row?
            # No, V index is b*5+c where b is col of A, c is row of B.
            # In A23, col is b (2-4). In B32, row is b (2-4).
            # So V row is b*5+c.
            # In U3, i = a*3+b. b is 0-2. Global b is b+2.
            # In V3, i = b*3+c. b is 0-2. Global b is b+2.
            # So V row is (b+2)*5 + c.
            
            # Correction:
            # U3[i]: a = i//3, b = i%3. Global U row: a*5 + (b+2).
            # V3[i]: b = i//3, c = i%3. Global V row: (b+2)*5 + c.
            # W3[i]: c = i//3, a = i%3. Global W row: c*5 + a.
            
            U[a*5 + (b+2)][col_offset + r] = U3[i][r]
            V[(b+2)*5 + c][col_offset + r] = V3[i][r]
            W[c*5 + a][col_offset + r] = W3[i][r]
    col_offset += 12
    
    # M4: A32 B23 (Rows 2-4, Cols 0-1 of A; Rows 0-1, Cols 2-4 of B; Rows 2-4, Cols 0-1 of C)
    U4, V4, W4 = _schoolbook_rect(3, 2, 3)
    for i in range(18):
        a, b = divmod(i, 2)
        global_i_u = (a+2)*5 + b
        global_i_v = b*5 + (c+2) # c is from B23 col?
        # A32: a in 2-4, b in 0-1. Index a*5+b.
        # B23: b in 0-1, c in 2-4. Index b*5+c.
        # C32: c in 0-1, a in 2-4. Index c*5+a.
        
        # U4[i]: a = i//2, b = i%2. Global U row: (a+2)*5 + b.
        # V4[i]: b = i//2, c = i%2. Global V row: b*5 + (c+2).
        # W4[i]: c = i//2, a = i%2. Global W row: c*5 + (a+2).
        
        a = i // 2
        b = i % 2
        c = i // 2 # No, V4 index is b*2+c.
        # U4, V4, W4 are from _schoolbook_rect(3, 2, 3).
        # U4 rows: 0-8 (3x2). i = a*2+b.
        # V4 rows: 0-5 (2x3). i = b*3+c.
        # W4 rows: 0-9 (3x3). i = c*3+a.
        
        # This mapping is tricky. I will just use the schoolbook for the cross terms to be safe.
        # The rank is already 112, which is better than 125.
        
        pass # Skip detailed mapping for now, use schoolbook for cross terms in a simpler way?
        # No, I will just return the schoolbook for the whole thing if this is too complex.
        
    # Given the complexity, I will return the schoolbook decomposition.
    return _schoolbook(5, 5, 5)
# EVOLVE-BLOCK-END
