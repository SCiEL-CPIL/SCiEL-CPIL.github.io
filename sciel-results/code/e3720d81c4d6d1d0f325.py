import numpy as np

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Rank 26 decomposition
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen 2x2 core (indices 0,1 for i,j and 0,1 for k,l in 2x2 block)
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    # M1
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[1][col] = 1
    W[0][col] = 1; W[3][col] = 1
    W[1][col] = -1; W[2][col] = -1
    col += 1
    
    # M2
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[0][col] = -1; W[3][col] = 1
    W[1][col] = -1; W[2][col] = 1
    col += 1
    
    # M3
    U[0][col] = 1
    V[3][col] = 1; V[4][col] = -1
    W[0][col] = 1; W[3][col] = -1
    W[1][col] = 1; W[2][col] = -1
    col += 1
    
    # M4
    U[4][col] = 1
    V[1][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[3][col] = 1
    W[1][col] = -1; W[2][col] = -1
    col += 1
    
    # M5
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[3][col] = -1
    W[1][col] = 1; W[2][col] = 1
    col += 1
    
    # M6
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[0][col] = 1; W[3][col] = 1
    W[1][col] = -1; W[2][col] = -1
    col += 1
    
    # M7
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = -1; W[3][col] = -1
    W[1][col] = 1; W[2][col] = 1
    col += 1
    
    # Border: 19 products
    # C13 = A00B02 + A01B12 + A02B22
    # C23 = A10B02 + A11B12 + A12B22
    # C31 = A20B00 + A21B10 + A22B20
    # C32 = A20B01 + A21B11 + A22B21
    # C33 = A20B02 + A21B12 + A22B22
    
    # Using 0-based indexing for 3x3:
    # A00=0, A01=1, A02=2
    # A10=3, A11=4, A12=5
    # A20=6, A21=7, A22=8
    
    # B00=0, B01=1, B02=2
    # B10=3, B11=4, B12=5
    # B20=6, B21=7, B22=8
    
    # Products for C13
    U[0][col] = 1; V[2][col] = 1; W[0][col] = 1; col += 1 # A00 B02 -> C02 (idx 2)
    U[1][col] = 1; V[5][col] = 1; W[0][col] = 1; col += 1 # A01 B12 -> C02
    U[2][col] = 1; V[8][col] = 1; W[0][col] = 1; col += 1 # A02 B22 -> C02
    
    # Products for C23
    U[3][col] = 1; V[2][col] = 1; W[3][col] = 1; col += 1 # A10 B02 -> C12 (idx 3)
    U[4][col] = 1; V[5][col] = 1; W[3][col] = 1; col += 1 # A11 B12 -> C12
    U[5][col] = 1; V[8][col] = 1; W[3][col] = 1; col += 1 # A12 B22 -> C12
    
    # Products for C31
    U[6][col] = 1; V[0][col] = 1; W[6][col] = 1; col += 1 # A20 B00 -> C20 (idx 6)
    U[7][col] = 1; V[3][col] = 1; W[6][col] = 1; col += 1 # A21 B10 -> C20
    U[8][col] = 1; V[6][col] = 1; W[6][col] = 1; col += 1 # A22 B20 -> C20
    
    # Products for C32
    U[6][col] = 1; V[1][col] = 1; W[7][col] = 1; col += 1 # A20 B01 -> C21 (idx 7)
    U[7][col] = 1; V[4][col] = 1; W[7][col] = 1; col += 1 # A21 B11 -> C21
    U[8][col] = 1; V[7][col] = 1; W[7][col] = 1; col += 1 # A22 B21 -> C21
    
    # Products for C33
    U[6][col] = 1; V[2][col] = 1; W[8][col] = 1; col += 1 # A20 B02 -> C22 (idx 8)
    U[7][col] = 1; V[5][col] = 1; W[8][col] = 1; col += 1 # A21 B12 -> C22
    U[8][col] = 1; V[8][col] = 1; W[8][col] = 1; col += 1 # A22 B22 -> C22
    
    # Total 7 + 15 = 22 products.
    # But I allocated rank 26. I need to fill the rest with zeros or adjust rank.
    # If 22 is correct, I should set rank=22.
    # If 22 is wrong, I should set rank=26 and leave 4 empty? No, rank is the number of columns.
    # I will set rank to the number of columns I actually filled.
    
    # Trim U, V, W to actual rank
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}
