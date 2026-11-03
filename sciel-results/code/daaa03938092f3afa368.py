import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_L B_11 (Strassen 2x2x2)
    # A_L indices: i in {0,1}, j in {0,1}
    # B_11 indices: j in {0,1}, k in {0,1}
    # C indices: i in {0,1}, k in {0,1}
    
    # M1 = (A00 + A11)(B00 + B11)
    # A coeffs: A00:1, A11:1
    # B coeffs: B00:1, B11:1
    # C coeffs: C00:1, C11:1
    U[0*3+0][col] = 1; U[1*3+1][col] = 1
    V[0*3+0][col] = 1; V[1*3+1][col] = 1
    W[0*2+0][col] = 1; W[1*2+1][col] = 1
    col += 1
    
    # M2 = (A10 + A11) B00
    # A coeffs: A10:1, A11:1
    # B coeffs: B00:1
    # C coeffs: C10:1
    U[1*3+0][col] = 1; U[1*3+1][col] = 1
    V[0*3+0][col] = 1
    W[0*2+1][col] = 1
    col += 1
    
    # M3 = A00 (B01 - B11)
    # A coeffs: A00:1
    # B coeffs: B01:1, B11:-1
    # C coeffs: C01:1
    U[0*3+0][col] = 1
    V[0*3+1][col] = 1; V[1*3+1][col] = -1
    W[1*2+0][col] = 1
    col += 1
    
    # M4 = A11 (B10 - B00)
    # A coeffs: A11:1
    # B coeffs: B10:1, B00:-1
    # C coeffs: C00:1, C10:1
    U[1*3+1][col] = 1
    V[1*3+0][col] = 1; V[0*3+0][col] = -1
    W[0*2+0][col] = 1; W[0*2+1][col] = 1
    col += 1
    
    # M5 = (A00 + A01) B11
    # A coeffs: A00:1, A01:1
    # B coeffs: B11:1
    # C coeffs: C00:-1, C01:1
    U[0*3+0][col] = 1; U[0*3+1][col] = 1
    V[1*3+1][col] = 1
    W[0*2+0][col] = -1; W[1*2+0][col] = 1
    col += 1
    
    # M6 = (A10 - A00) (B00 + B01)
    # A coeffs: A10:1, A00:-1
    # B coeffs: B00:1, B01:1
    # C coeffs: C11:1
    U[1*3+0][col] = 1; U[0*3+0][col] = -1
    V[0*3+0][col] = 1; V[0*3+1][col] = 1
    W[1*2+1][col] = 1
    col += 1
    
    # M7 = (A01 - A11) (B10 + B11)
    # A coeffs: A01:1, A11:-1
    # B coeffs: B10:1, B11:1
    # C coeffs: C00:1, C11:1
    U[0*3+1][col] = 1; U[1*3+1][col] = -1
    V[1*3+0][col] = 1; V[1*3+1][col] = 1
    W[0*2+0][col] = 1; W[1*2+1][col] = 1
    col += 1
    
    # Part 2: A_L B_12 (Schoolbook 2x2x1)
    # A_L: i in {0,1}, j in {0,1}
    # B_12: j in {0,1}, k=2
    # C: i in {0,1}, k=2
    
    # C02 += A00 B02
    U[0*3+0][col] = 1
    V[0*3+2][col] = 1
    W[2*2+0][col] = 1
    col += 1
    
    # C02 += A01 B12
    U[0*3+1][col] = 1
    V[1*3+2][col] = 1
    W[2*2+0][col] = 1
    col += 1
    
    # C12 += A10 B02
    U[1*3+0][col] = 1
    V[0*3+2][col] = 1
    W[2*2+1][col] = 1
    col += 1
    
    # C12 += A11 B12
    U[1*3+1][col] = 1
    V[1*3+2][col] = 1
    W[2*2+1][col] = 1
    col += 1
    
    # Part 3: A_R B_B (Schoolbook 2x1x3)
    # A_R: i in {0,1}, j=2
    # B_B: j=2, k in {0,1,2}
    # C: i in {0,1}, k in {0,1,2}
    
    # C00 += A02 B20
    U[0*3+2][col] = 1
    V[2*3+0][col] = 1
    W[0*2+0][col] = 1
    col += 1
    
    # C01 += A02 B21
    U[0*3+2][col] = 1
    V[2*3+1][col] = 1
    W[1*2+0][col] = 1
    col += 1
    
    # C02 += A02 B22
    U[0*3+2][col] = 1
    V[2*3+2][col] = 1
    W[2*2+0][col] = 1
    col += 1
    
    # C10 += A12 B20
    U[1*3+2][col] = 1
    V[2*3+0][col] = 1
    W[0*2+1][col] = 1
    col += 1
    
    # C11 += A12 B21
    U[1*3+2][col] = 1
    V[2*3+1][col] = 1
    W[1*2+1][col] = 1
    col += 1
    
    # C12 += A12 B22
    U[1*3+2][col] = 1
    V[2*3+2][col] = 1
    W[2*2+1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}

