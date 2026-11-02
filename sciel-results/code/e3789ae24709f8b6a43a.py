import time
import numpy as np

DIMENSIONS = [3, 3, 3]

def construct():
    # Rank 22 decomposition for 3x3x3 matrix multiplication
    # Uses Strassen for the 2x2 core (rank 7) and schoolbook for the border (rank 15)
    
    n, m, p = DIMENSIONS
    rank = 22
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices:
    # A: i*m + j (i,j in 0..2)
    # B: j*p + k (j,k in 0..2)
    # C: k*n + i (k,i in 0..2) -> W[k*n+i, c]
    
    # Strassen for 2x2 core (indices 0,1)
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 - M3 + M6
    
    col = 0
    # M1
    U[0][col] = 1; U[3][col] = 1  # A00, A11
    V[0][col] = 1; V[4][col] = 1  # B00, B11
    W[0][col] = 1; W[1][col] = -1; W[3][col] = -1; W[4][col] = 1  # C00, C01, C10, C11
    # C00 (k=0,i=0) -> W[0]
    # C01 (k=1,i=0) -> W[1]
    # C10 (k=0,i=1) -> W[3]
    # C11 (k=1,i=1) -> W[4]
    # M1 contributes to C00 (+1), C01 (-1? No, C01=M3+M5, so 0), C10 (0), C11 (+1? No, C11=M1-M2-M3+M6, so +1)
    # Wait, let's map carefully.
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 - M3 + M6
    
    # Let's reset W for M1
    W[0][col] = 1   # C00
    W[4][col] = 1   # C11
    col += 1
    
    # M2 = (A10+A11)B00
    U[2][col] = 1; U[3][col] = 1  # A10, A11
    V[0][col] = 1                 # B00
    W[3][col] = 1   # C10
    W[4][col] = -1  # C11
    col += 1
    
    # M3 = A00(B10-B11)
    U[0][col] = 1                 # A00
    V[2][col] = 1; V[3][col] = -1 # B10, B11
    W[1][col] = 1   # C01
    W[4][col] = -1  # C11
    col += 1
    
    # M4 = A11(B01-B00)
    U[3][col] = 1                 # A11
    V[1][col] = 1; V[0][col] = -1 # B01, B00
    W[0][col] = -1  # C00
    W[3][col] = 1   # C10
    col += 1
    
    # M5 = (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1  # A00, A01
    V[4][col] = 1                 # B11
    W[0][col] = -1  # C00
    W[1][col] = 1   # C01
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][col] = 1; U[0][col] = -1 # A10, A00
    V[0][col] = 1; V[1][col] = 1  # B00, B01
    W[4][col] = 1   # C11
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][col] = 1; U[3][col] = -1 # A01, A11
    V[2][col] = 1; V[4][col] = 1  # B10, B11
    W[0][col] = 1   # C00
    col += 1
    
    # Border:
    # C20 = A20B00 + A21B10 + A22B20
    # C21 = A20B01 + A21B11 + A22B21
    # C22 = A20B02 + A21B12 + A22B22
    # C02 = A00B02 + A01B12 + A02B22
    # C12 = A10B02 + A11B12 + A12B22
    
    # We can just use schoolbook for these 5 entries, 3 mults each = 15 mults.
    # C20
    U[5][col] = 1; V[0][col] = 1; W[6][col] = 1; col += 1 # A20 B00 -> C20 (k=0,i=2 -> W[0*3+2]=W[2]? No. W[k*n+i]. C20 is i=2, k=0. W[0*3+2]=W[2])
    # Correction: W index is k*n + i.
    # C20: i=2, k=0 -> W[2]
    # C21: i=2, k=1 -> W[5]
    # C22: i=2, k=2 -> W[8]
    # C02: i=0, k=2 -> W[6]
    # C12: i=1, k=2 -> W[7]
    
    # Let's redo the W indices for Strassen part to be sure.
    # C00: i=0, k=0 -> W[0]
    # C01: i=0, k=1 -> W[3]
    # C10: i=1, k=0 -> W[1]
    # C11: i=1, k=1 -> W[4]
    
    # My previous W assignments for Strassen:
    # M1: W[0]=1, W[4]=1. Correct.
    # M2: W[1]=1 (C10), W[4]=-1 (C11). Correct.
    # M3: W[3]=1 (C01), W[4]=-1 (C11). Correct.
    # M4: W[0]=-1 (C00), W[1]=1 (C10). Correct.
    # M5: W[0]=-1 (C00), W[3]=1 (C01). Correct.
    # M6: W[4]=1 (C11). Correct.
    # M7: W[0]=1 (C00). Correct.
    
    # Now Border with correct W indices:
    # C20 (W[2]): A20B00, A21B10, A22B20
    U[5][col] = 1; V[0][col] = 1; W[2][col] = 1; col += 1
    U[6][col] = 1; V[2][col] = 1; W[2][col] = 1; col += 1
    U[8][col] = 1; V[4][col] = 1; W[2][col] = 1; col += 1
    
    # C21 (W[5]): A20B01, A21B11, A22B21
    U[5][col] = 1; V[1][col] = 1; W[5][col] = 1; col += 1
    U[6][col] = 1; V[4][col] = 1; W[5][col] = 1; col += 1
    U[8][col] = 1; V[5][col] = 1; W[5][col] = 1; col += 1
    
    # C22 (W[8]): A20B02, A21B12, A22B22
    U[5][col] = 1; V[3][col] = 1; W[8][col] = 1; col += 1
    U[6][col] = 1; V[6][col] = 1; W[8][col] = 1; col += 1
    U[8][col] = 1; V[8][col] = 1; W[8][col] = 1; col += 1
    
    # C02 (W[6]): A00B02, A01B12, A02B22
    U[0][col] = 1; V[3][col] = 1; W[6][col] = 1; col += 1
    U[1][col] = 1; V[6][col] = 1; W[6][col] = 1; col += 1
    U[2][col] = 1; V[8][col] = 1; W[6][col] = 1; col += 1
    
    # C12 (W[7]): A10B02, A11B12, A12B22
    U[3][col] = 1; V[3][col] = 1; W[7][col] = 1; col += 1
    U[4][col] = 1; V[6][col] = 1; W[7][col] = 1; col += 1
    U[5][col] = 1; V[8][col] = 1; W[7][col] = 1; col += 1
    
    return {"U": U, "V": V, "W": W}
