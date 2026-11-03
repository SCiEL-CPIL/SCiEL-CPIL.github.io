import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is n*m*p
    rank_school = n * m * p
    
    # We will use 1 Strassen block for I={0,1}, J={0,1}, K={0,1}
    # This saves 1 multiplication (8 -> 7)
    # Total rank = 75 - 1 = 74
    
    rank = rank_school - 1
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define Strassen factors for the 2x2x2 block
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # A indices: i*m + j
    # B indices: j*p + k
    # C indices: k*n + i
    
    # M1: (A00 + A11)(B00 + B11)
    U[0*m+0][col] = 1
    U[1*m+1][col] = 1
    V[0*p+0][col] = 1
    V[1*p+1][col] = 1
    W[0*n+0][col] = 1  # C00
    W[1*n+1][col] = 1  # C11
    col += 1
    
    # M2: (A10 + A11)B00
    U[1*m+0][col] = 1
    U[1*m+1][col] = 1
    V[0*p+0][col] = 1
    W[1*n+0][col] = 1  # C10
    col += 1
    
    # M3: A00(B10 - B11)
    U[0*m+0][col] = 1
    V[1*p+0][col] = 1
    V[1*p+1][col] = -1
    W[0*n+1][col] = 1  # C01
    col += 1
    
    # M4: A11(B01 - B00)
    U[1*m+1][col] = 1
    V[0*p+1][col] = 1
    V[0*p+0][col] = -1
    W[0*n+0][col] = 1  # C00
    W[1*n+0][col] = 1  # C10
    col += 1
    
    # M5: (A00 + A01)B11
    U[0*m+0][col] = 1
    U[0*m+1][col] = 1
    V[1*p+1][col] = 1
    W[0*n+0][col] = -1 # C00
    W[0*n+1][col] = 1  # C01
    col += 1
    
    # M6: (A10 - A00)(B00 + B01)
    U[1*m+0][col] = 1
    U[0*m+0][col] = -1
    V[0*p+0][col] = 1
    V[0*p+1][col] = 1
    W[1*n+1][col] = 1  # C11
    col += 1
    
    # M7: (A01 - A11)(B10 + B11)
    U[0*m+1][col] = 1
    U[1*m+1][col] = -1
    V[1*p+0][col] = 1
    V[1*p+1][col] = 1
    W[0*n+0][col] = 1  # C00
    W[1*n+1][col] = -1 # C11
    col += 1
    
    # Now add the remaining schoolbook terms
    # We must skip the 8 terms covered by Strassen
    # Covered: i in {0,1}, j in {0,1}, k in {0,1}
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if in Strassen block
                if i in [0, 1] and j in [0, 1] and k in [0, 1]:
                    continue
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
