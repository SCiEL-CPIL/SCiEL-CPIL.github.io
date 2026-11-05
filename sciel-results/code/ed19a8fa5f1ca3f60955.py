import time
import sys

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen decomposition for 3x3x3
    # Indices for A (3x3): 0..8
    # Indices for B (3x3): 0..8
    # Indices for C (3x3): 0..8
    
    # Term 1: (A00+A22)(B00+B22)
    U[0][0] += 1
    U[8][0] += 1
    V[0][0] += 1
    V[8][0] += 1
    W[0][0] += 1   # C00
    W[3][0] += 1   # C10
    W[5][0] += 1   # C12
    W[6][0] -= 1   # C20
    W[7][0] += 1   # C21
    W[8][0] -= 1   # C22
    
    # Term 2: (A10+A20)B02
    U[4][1] += 1
    U[6][1] += 1
    V[2][1] += 1
    W[2][1] += 1   # C02
    W[3][1] -= 1   # C10
    W[4][1] += 1   # C11
    W[5][1] -= 1   # C12
    W[7][1] += 1   # C21
    
    # Term 3: A01(B10-B20)
    U[1][2] += 1
    V[6][2] += 1
    V[12][2] -= 1
    W[1][2] += 1   # C01
    W[4][2] -= 1   # C11
    W[6][2] += 1   # C20
    W[7][2] -= 1   # C21
    
    # Term 4: A21(B01-B11)
    U[7][3] += 1
    V[1][3] += 1
    V[7][3] -= 1
    W[0][3] += 1   # C00
    W[3][3] -= 1   # C10
    W[4][3] += 1   # C11
    W[7][3] -= 1   # C21
    W[8][3] += 1   # C22
    
    # Term 5: (A00+A10)B11
    U[0][4] += 1
    U[4][4] += 1
    V[7][4] += 1
    W[0][4] -= 1   # C00
    W[2][4] += 1   # C02
    W[4][4] += 1   # C11
    W[6][4] += 1   # C20
    
    # Term 6: (A11-A01)(B00+B01)
    U[5][5] += 1
    U[1][5] -= 1
    V[0][5] += 1
    V[1][5] += 1
    W[3][5] += 1   # C10
    W[5][5] -= 1   # C12
    W[6][5] -= 1   # C20
    W[8][5] += 1   # C22
    
    # Term 7: (A11-A22)(B22+B12)
    U[5][6] += 1
    U[8][6] -= 1
    V[8][6] += 1
    V[14][6] += 1
    W[0][6] -= 1   # C00
    W[1][6] += 1   # C01
    W[5][6] += 1   # C12
    W[6][6] -= 1   # C20
    W[8][6] += 1   # C22
    
    # Copy the first 7 columns to the next 7 columns for U
    # For V and W, we need to shift rows.
    
    for c in range(7):
        # U
        for r in range(n * m):
            U[r][c + 7] = U[r][c]
            
        # V: rows 0-8 are for B cols 0-2, rows 9-17 are for B cols 3-5
        # For the second block (c+7), the B indices shift by 3 (m*p? no, B is 3x6, flattened 6j+k)
        # B index j*p+k. For cols 3-5, k is 3,4,5.
        # In the first block, k was 0,1,2.
        # So V row index for B[j, k+3] is j*p + k + 3.
        # V row index for B[j, k] is j*p + k.
        # So we add 3 to the row index?
        # Wait, V is (m*p) x rank. m=3, p=6. Size 18.
        # B[j, k] index is j*6 + k.
        # First block uses k=0,1,2.
        # Second block uses k=3,4,5.
        # So yes, row index increases by 3.
        
        for r in range(m * p):
            if r < 9:
                V[r][c + 7] = 0
            else:
                # Map r (9..17) to r-9 (0..8) of the first block V
                V[r][c + 7] = V[r - 9][c]
                
        # W: rows 0-8 are for C cols 0-2, rows 9-17 are for C cols 3-5
        # C index i*p + k.
        # First block k=0,1,2.
        # Second block k=3,4,5.
        # So row index increases by 3.
        
        for r in range(p * n):
            if r < 9:
                W[r][c + 7] = 0
            else:
                # Map r (9..17) to r-9 (0..8) of the first block W
                W[r][c + 7] = W[r - 9][c]

    return {"U": U, "V": V, "W": W}
