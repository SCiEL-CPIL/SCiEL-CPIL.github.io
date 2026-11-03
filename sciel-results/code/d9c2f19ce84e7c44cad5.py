import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Rank 17 construction
    # 7 Strassen terms for A[0:2, 0:2] * B[0:2, 0:2]
    # 10 Schoolbook terms for the rest
    
    U = []
    V = []
    W = []
    
    # Strassen terms
    # M1 = (a11+a22)(b11+b22)
    U.append([1, 0, 0, 1, 0, 0]) # a11, a12, a13, a21, a22, a23
    V.append([1, 1, 0, 1, 1, 0]) # b11, b12, b13, b21, b22, b23, b31, b32, b33? 
    # Wait, V dimension is m*p = 3*3 = 9.
    # U dimension is n*m = 2*3 = 6.
    # W dimension is p*n = 3*2 = 6.
    
    # Let's define indices clearly.
    # U rows: (i,j) where i in 0..1, j in 0..2.
    # V rows: (j,k) where j in 0..2, k in 0..2.
    # W rows: (k,i) where k in 0..2, i in 0..1.
    
    U = [[0]*17 for _ in range(6)]
    V = [[0]*17 for _ in range(9)]
    W = [[0]*17 for _ in range(6)]
    
    col = 0
    
    # Strassen 1: (a11+a22)(b11+b22)
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[4][col] = 1
    # Result: a11b11 + a11b22 + a22b11 + a22b22
    # W contribution:
    # C11 gets a11b11 + a22b22 ? No, Strassen combination is done in W.
    # W[k*n+i, col]
    # This term contributes to C11 (i=0, k=0) and C12 (i=0, k=1) and C21 (i=1, k=0) and C22 (i=1, k=1).
    # Actually, the Strassen algorithm computes the 4 entries of the 2x2 product.
    # We need to map the 7 products to the 4 output entries.
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B22-B11)
    # M4 = A22(B11-B22)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We will add these 7 terms.
    # Term 1
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[1][col] = 0; W[2][col] = 0; W[3][col] = 1 # C11, C22
    # Wait, W index is k*n+i.
    # C11 is i=0, k=0 -> index 0.
    # C12 is i=0, k=1 -> index 1.
    # C21 is i=1, k=0 -> index 2.
    # C22 is i=1, k=1 -> index 3.
    # M1 contributes to C11 and C22.
    W[0][col] = 1
    W[3][col] = 1
    col += 1
    
    # Term 2: (A21+A22)B11
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[2][col] = 1 # C21
    col += 1
    
    # Term 3: A11(B22-B11)
    U[0][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[1][col] = 1 # C12
    col += 1
    
    # Term 4: A22(B11-B22)
    U[4][col] = 1
    V[0][col] = 1; V[4][col] = -1
    W[0][col] = 1; W[2][col] = 1 # C11, C21
    col += 1
    
    # Term 5: (A11+A12)B22
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[1][col] = 1 # C11, C12
    col += 1
    
    # Term 6: (A21-A11)(B11+B12)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1 # C22
    col += 1
    
    # Term 7: (A12-A22)(B21+B22)
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1 # C11
    col += 1
    
    # Now add the 10 schoolbook terms for the rest.
    # Remaining products:
    # C11: a13b31
    # C12: a13b32
    # C13: a11b13, a12b23, a13b33
    # C21: a23b31
    # C22: a23b32
    # C23: a21b13, a22b23, a23b33
    
    # T8: a13b31 -> C11
    U[2][col] = 1
    V[6][col] = 1
    W[0][col] = 1
    col += 1
    
    # T9: a13b32 -> C12
    U[2][col] = 1
    V[7][col] = 1
    W[1][col] = 1
    col += 1
    
    # T10: a23b31 -> C21
    U[5][col] = 1
    V[6][col] = 1
    W[2][col] = 1
    col += 1
    
    # T11: a23b32 -> C22
    U[5][col] = 1
    V[7][col] = 1
    W[3][col] = 1
    col += 1
    
    # T12: a11b13 -> C13
    U[0][col] = 1
    V[2][col] = 1
    W[4][col] = 1 # k=2, i=0 -> index 4
    col += 1
    
    # T13: a12b23 -> C13
    U[1][col] = 1
    V[5][col] = 1
    W[4][col] = 1
    col += 1
    
    # T14: a13b33 -> C13
    U[2][col] = 1
    V[8][col] = 1
    W[4][col] = 1
    col += 1
    
    # T15: a21b13 -> C23
    U[3][col] = 1
    V[2][col] = 1
    W[5][col] = 1 # k=2, i=1 -> index 5
    col += 1
    
    # T16: a22b23 -> C23
    U[4][col] = 1
    V[5][col] = 1
    W[5][col] = 1
    col += 1
    
    # T17: a23b33 -> C23
    U[5][col] = 1
    V[8][col] = 1
    W[5][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
