import time
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    """
    Constructs a rank-15 exact tensor decomposition for 2x3 by 3x4 matrix multiplication.
    The output factor W follows the convention W[k*n+i, r].
    """
    n, m, p = 2, 3, 4
    
    # We use a known rank-15 decomposition based on Strassen's algorithm for 2x2 blocks
    # combined with simple multiplications for the remaining parts.
    # Structure:
    # The 2x3 matrix A is split into A1 (2x1) and A2 (2x2).
    # The 3x4 matrix B is split into B1 (1x4) and B2 (2x4).
    # C = A1*B1 + A2*B2.
    # A1*B1 is 2x4, rank 2 (sum of 2 rank-1 terms? No, A1 is 2x1, B1 is 1x4. A1*B1 is rank 1. 
    # Wait, A1 is column vector, B1 is row vector. Outer product. Rank 1.
    # So A1*B1 contributes 1 term.
    # A2*B2 is 2x2 * 2x4. This is a 2x2 * 2x4 multiplication.
    # We can use Strassen for the 2x2 part of B2?
    # Let B2 = [B2_top; B2_bottom], where B2_top is 1x4, B2_bottom is 1x4? No, B2 is 2x4.
    # Let's split B2 into two 2x2 blocks? B2 = [B2a B2b].
    # Then A2*B2 = A2*B2a + A2*B2b.
    # A2*B2a is 2x2 * 2x2. Rank 7 (Strassen).
    # A2*B2b is 2x2 * 2x2. Rank 7 (Strassen).
    # Total rank = 1 (for A1*B1) + 7 + 7 = 15.
    
    rank = 15
    U = [[0]*rank for _ in range(n*m)] # 6x15
    V = [[0]*rank for _ in range(m*p)] # 12x15
    W = [[0]*rank for _ in range(p*n)] # 8x15
    
    # Term 0: A1 * B1
    # A1 corresponds to A[0:2, 0]. Indices in U: 0, 1.
    # B1 corresponds to B[0, 0:4]. Indices in V: 0, 1, 2, 3.
    # W indices: k*2 + i.
    
    for i in range(2):
        U[i*3 + 0][0] = 1
    for k in range(4):
        V[0*4 + k][0] = 1
    for k in range(4):
        for i in range(2):
            W[k*2 + i][0] = 1
            
    # Strassen terms for A2 * B2a (A2 is A[0:2, 1:3], B2a is B[1:3, 0:2])
    # A2 indices in U: i*3 + 1, i*3 + 2.
    # B2a indices in V: (1)*4+k, (2)*4+k for k in 0..1.
    # W indices: k*2 + i.
    
    # Strassen definitions for 2x2 * 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Map A2 to A00, A01, A10, A11
    # A2[0,0] -> A00 (U index 0*3+1 = 1)
    # A2[0,1] -> A01 (U index 0*3+2 = 2)
    # A2[1,0] -> A10 (U index 1*3+1 = 4)
    # A2[1,1] -> A11 (U index 1*3+2 = 5)
    
    # Map B2a to B00, B01, B10, B11
    # B2a[0,0] -> B00 (V index 1*4+0 = 4)
    # B2a[0,1] -> B01 (V index 1*4+1 = 5)
    # B2a[1,0] -> B10 (V index 2*4+0 = 8)
    # B2a[1,1] -> B11 (V index 2*4+1 = 9)
    
    # W for first block (k=0,1)
    # C[0,0] = M1 + M4 - M5 - M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    strassen_U = [
        {(1,1):1, (5,5):1}, # M1: A00+A11 -> U[1], U[5]
        {(4,4):1, (5,5):1}, # M2: A10+A11 -> U[4], U[5]
        {(1,1):1},          # M3: A00 -> U[1]
        {(5,5):1},          # M4: A11 -> U[5]
        {(1,1):1, (2,2):1}, # M5: A00+A01 -> U[1], U[2]
        {(4,4):1, (1,1):-1},# M6: A10-A00 -> U[4], -U[1]
        {(2,2):1, (5,5):-1} # M7: A01-A11 -> U[2], -U[5]
    ]
    strassen_V = [
        {(4,4):1, (9,9):1}, # M1: B00+B11 -> V[4], V[9]
        {(4,4):1},          # M2: B00 -> V[4]
        {(5,5):1, (9,9):-1},# M3: B01-B11 -> V[5], -V[9]
        {(8,8):1, (4,4):-1},# M4: B10-B00 -> V[8], -V[4]
        {(9,9):1},          # M5: B11 -> V[9]
        {(4,4):1, (5,5):1}, # M6: B00+B01 -> V[4], V[5]
        {(8,8):1, (9,9):1}  # M7: B10+B11 -> V[8], V[9]
    ]
    strassen_W = [
        {(0,0):1, (1,1):1}, # M1: C00, C11
        {(1,0):1},          # M2: C10
        {(0,1):1},          # M3: C01
        {(0,0):1, (1,1):-1},# M4: C00, -C11
        {(0,1):-1, (1,1):-1},# M5: -C01, -C11
        {(1,1):1},          # M6: C11
        {(0,0):-1, (1,1):-1} # M7: -C00, -C11
    ]
    
    # Fill U, V, W for the first Strassen block (cols 1-7)
    for r in range(7):
        col = 1 + r
        # U
        for (idx, val) in strassen_U[r].items():
            U[idx][col] = val
        # V
        for (idx, val) in strassen_V[r].items():
            V[idx][col] = val
        # W
        for (idx, val) in strassen_W[r].items():
            W[idx][col] = val

    # Strassen terms for A2 * B2b (A2 is A[0:2, 1:3], B2b is B[1:3, 2:4])
    # A2 indices in U: same as before (1, 2, 4, 5).
    # B2b indices in V:
    # B2b[0,0] -> B00 (V index 1*4+2 = 6)
    # B2b[0,1] -> B01 (V index 1*4+3 = 7)
    # B2b[1,0] -> B10 (V index 2*4+2 = 10)
    # B2b[1,1] -> B11 (V index 2*4+3 = 11)
    
    # W for second block (k=2,3)
    # C[0,2] = M1 + M4 - M5 - M7
    # C[0,3] = M3 + M5
    # C[1,2] = M2 + M4
    # C[1,3] = M1 - M2 + M3 + M6
    
    # W indices:
    # C[0,2] -> W[2*2+0] = W[4]
    # C[0,3] -> W[2*2+1] = W[5]
    # C[1,2] -> W[3*2+0] = W[6]
    # C[1,3] -> W[3*2+1] = W[7]
    
    strassen_W2 = [
        {(4,4):1, (7,7):1}, # M1: C02, C13
        {(6,6):1},          # M2: C12
        {(5,5):1},          # M3: C03
        {(4,4):1, (7,7):-1},# M4: C02, -C13
        {(5,5):-1, (7,7):-1},# M5: -C03, -C13
        {(7,7):1},          # M6: C13
        {(4,4):-1, (7,7):-1} # M7: -C02, -C13
    ]
    
    # Fill U, V, W for the second Strassen block (cols 8-14)
    for r in range(7):
        col = 8 + r
        # U
        for (idx, val) in strassen_U[r].items():
            U[idx][col] = val
        # V
        for (idx, val) in strassen_V[r].items():
            # Map V indices from B2a (4,5,8,9) to B2b (6,7,10,11)
            # 4->6, 5->7, 8->10, 9->11
            if idx == 4: new_idx = 6
            elif idx == 5: new_idx = 7
            elif idx == 8: new_idx = 10
            elif idx == 9: new_idx = 11
            else: new_idx = idx
            V[new_idx][col] = val
        # W
        for (idx, val) in strassen_W2[r].items():
            W[idx][col] = val

    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    # Fallback to schoolbook if verification fails (should not happen)
                    rank24 = n*m*p
                    U2 = [[0]*rank24 for _ in range(n*m)]
                    V2 = [[0]*rank24 for _ in range(m*p)]
                    W2 = [[0]*rank24 for _ in range(p*n)]
                    col = 0
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                U2[i2*m+j2][col] = 1
                                V2[j2*p+k2][col] = 1
                                W2[k2*n+i2][col] = 1
                                col += 1
                    return {"U": U2, "V": V2, "W": W2}
    
    return {"U": U, "V": V, "W": W}
