import time
import numpy as np

DIMENSIONS = [2, 2, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=5
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (10, rank)
    # W: (p*n, rank) = (10, rank)
    
    rank = 19
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for U: i*m + j
    # i in 0..1, j in 0..1
    # 0: A00, 1: A01, 2: A10, 3: A11
    
    # Indices for V: j*p + k
    # j in 0..1, k in 0..4
    # 0: B00, 1: B01, 2: B02, 3: B03, 4: B04
    # 5: B10, 6: B11, 7: B12, 8: B13, 9: B14
    
    # Indices for W: k*n + i
    # k in 0..4, i in 0..1
    # 0: C00, 1: C01, 2: C02, 3: C03, 4: C04
    # 5: C10, 6: C11, 7: C12, 8: C13, 9: C14
    
    # --- Part 1: Strassen for 2x2x2 (Cols 0-1 of B, Cols 0-1 of C) ---
    # Col 0: M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1 # B00 is idx 0, B11 is idx 6? 
    # Wait, V index j*p+k.
    # B00: j=0, k=0 -> 0.
    # B11: j=1, k=1 -> 1*5+1 = 6.
    # My previous manual trace used local indices for 2x2 block.
    # Here I must use global indices.
    
    # Let's re-map Strassen to global indices.
    # B block is cols 0,1.
    # B00 (global 0), B01 (global 1), B10 (global 5), B11 (global 6).
    
    # Term 0 (M1): (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[6][0] = 1
    # C part: C00 += 1, C11 += 1
    # C00: k=0, i=0 -> 0.
    # C11: k=1, i=1 -> 1*2+1 = 3.
    W[0][0] = 1; W[3][0] = 1
    
    # Term 1 (M2): (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # C part: C10 += 1, C11 -= 1
    # C10: k=0, i=1 -> 1.
    # C11: k=1, i=1 -> 3.
    W[1][1] = 1; W[3][1] = -1
    
    # Term 2 (M3): A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[6][2] = -1
    # C part: C01 += 1, C11 += 1
    # C01: k=1, i=0 -> 2.
    # C11: k=1, i=1 -> 3.
    W[2][2] = 1; W[3][2] = 1
    
    # Term 3 (M4): A11(B10-B00)
    U[3][3] = 1
    V[5][3] = 1; V[0][3] = -1
    # C part: C00 += 1, C10 += 1
    # C00: 0. C10: 1.
    W[0][3] = 1; W[1][3] = 1
    
    # Term 4 (M5): (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[6][4] = 1
    # C part: C00 -= 1, C01 += 1
    # C00: 0. C01: 2.
    W[0][4] = -1; W[2][4] = 1
    
    # Term 5 (M6): (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # C part: C11 += 1
    # C11: 3.
    W[3][5] = 1
    
    # Term 6 (M7): (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[5][6] = 1; V[6][6] = 1
    # C part: C00 += 1
    # C00: 0.
    W[0][6] = 1
    
    # --- Part 2: Schoolbook for Col 2 of B (Global Index 2, 7) ---
    # Cols 7-10
    # Term 7: A00 B02 -> C02
    U[0][7] = 1
    V[2][7] = 1 # B02: j=0, k=2 -> 2
    W[4][7] = 1 # C02: k=2, i=0 -> 4
    
    # Term 8: A01 B12 -> C02
    U[1][8] = 1
    V[7][8] = 1 # B12: j=1, k=2 -> 7
    W[4][8] = 1
    
    # Term 9: A10 B02 -> C12
    U[2][9] = 1
    V[2][9] = 1
    W[5][9] = 1 # C12: k=2, i=1 -> 5
    
    # Term 10: A11 B12 -> C12
    U[3][10] = 1
    V[7][10] = 1
    W[5][10] = 1
    
    # --- Part 3: Schoolbook for Col 3 of B (Global Index 3, 8) ---
    # Cols 11-14
    # Term 11: A00 B03 -> C03
    U[0][11] = 1
    V[3][11] = 1 # B03: 3
    W[6][11] = 1 # C03: 6
    
    # Term 12: A01 B13 -> C03
    U[1][12] = 1
    V[8][12] = 1 # B13: 8
    W[6][12] = 1
    
    # Term 13: A10 B03 -> C13
    U[2][13] = 1
    V[3][13] = 1
    W[7][13] = 1 # C13: 7
    
    # Term 14: A11 B13 -> C13
    U[3][14] = 1
    V[8][14] = 1
    W[7][14] = 1
    
    # --- Part 4: Schoolbook for Col 4 of B (Global Index 4, 9) ---
    # Cols 15-18
    # Term 15: A00 B04 -> C04
    U[0][15] = 1
    V[4][15] = 1 # B04: 4
    W[8][15] = 1 # C04: 8
    
    # Term 16: A01 B14 -> C04
    U[1][16] = 1
    V[9][16] = 1 # B14: 9
    W[8][16] = 1
    
    # Term 17: A10 B04 -> C14
    U[2][17] = 1
    V[4][17] = 1
    W[9][17] = 1 # C14: 9
    
    # Term 18: A11 B14 -> C14
    U[3][18] = 1
    V[9][18] = 1
    W[9][18] = 1
    
    return {"U": U, "V": V, "W": W}
