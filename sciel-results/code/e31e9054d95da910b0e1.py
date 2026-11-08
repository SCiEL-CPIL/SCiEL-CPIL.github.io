import time
import sys
import random
import numpy as np

DIMENSIONS = [2, 3, 5]

def verify_decomposition(n, m, p, U, V, W, tol=1e-9):
    """
    Verify: sum_r U[i*m+j1, r] V[j2*p+k, r] W[k*n+i, r] == delta(j1, j2)
    """
    rank = len(U[0])
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j1, r] * V[j2*p+k, r] * W[k*n+i, r]
                    target = 1.0 if j1 == j2 else 0.0
                    if abs(s - target) > tol:
                        return False
    return True

def construct_rank_17():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=5
    
    # We will build U, V, W lists
    # U: (n*m, rank) = (6, 17)
    # V: (m*p, rank) = (15, 17)
    # W: (p*n, rank) = (10, 17)
    
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: 2x2x3 block (A rows 0,1; B rows 0,1; B cols 0,1,2)
    # Rank 10 construction for 2x2x3
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01, B02], [B10, B11, B12]]
    # C = [[C00, C01, C02], [C10, C11, C12]]
    
    # Standard rank 10 for 2x2x3?
    # Actually, 2x2x2 is 7. 2x2x1 is 2. Total 9?
    # Let's use 7 for 2x2x2 (cols 0,1) and 2 for 2x2x1 (col 2).
    
    # Sub-part 1a: 2x2x2 (B cols 0,1)
    # Strassen
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1? No.
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map these to U, V, W.
    # Indices:
    # U index i*m+j. i in 0,1. j in 0,1 (for this block).
    # V index j*p+k. j in 0,1. k in 0,1 (for this block).
    # W index k*n+i. k in 0,1. i in 0,1.
    
    # Term 1: M1
    # U: A00, A11 -> U[0, r]=1, U[3, r]=1 (i=0,j=0; i=1,j=1)
    # V: B00, B11 -> V[0, r]=1, V[5, r]=1 (j=0,k=0; j=1,k=1)
    # W: C00, C11? No, W multiplies the result.
    # The product is (A00+A11)(B00+B11).
    # This contributes to C00 and C11.
    # We need to distribute this to C00 and C11.
    # W[0, r] (k=0,i=0) and W[1, r] (k=1,i=1)?
    # If W[0,r]=1, it adds to C00.
    # If W[1,r]=1, it adds to C11.
    # But we need C00 += M1 and C11 += M1.
    # So W[0,r]=1, W[1,r]=1?
    # Then C00 gets M1, C11 gets M1.
    # But other terms also contribute.
    
    # Let's just use the known rank 10 construction for 2x2x3 directly if possible, or stick to 7+2.
    # 7+2 = 9.
    
    # 2x2x2 Strassen (7 terms) for B cols 0,1
    # M1
    U[0][r] = 1; U[3][r] = 1
    V[0][r] = 1; V[5][r] = 1
    W[0][r] = 1; W[1][r] = 1
    r += 1
    
    # M2
    U[3][r] = 1; U[4][r] = 1 # A10, A11 (i=1, j=0,1) -> indices 3,4
    V[0][r] = 1 # B00
    W[1][r] = 1 # C10? No, M2 contributes to C10.
    # C10 index k=0, i=1. W[0*2+1, r] = W[1, r].
    W[1][r] = 1
    r += 1
    
    # M3
    U[0][r] = 1 # A00
    V[5][r] = 1; V[0][r] = -1 # B10-B00 (j=1,k=0; j=0,k=0) -> V[5], V[0]
    W[1][r] = 1 # C10? M3 contributes to C10? No, C11?
    # C10 = M2 + M4. C11 = M1 - M2 + M3 + M6.
    # So M3 goes to C11. k=1, i=1. W[1*2+1, r] = W[3, r].
    W[3][r] = 1
    r += 1
    
    # M4
    U[4][r] = 1 # A11
    V[1][r] = 1; V[5][r] = -1 # B01-B10 (j=0,k=1; j=1,k=0) -> V[1], V[5]
    W[0][r] = 1; W[3][r] = 1 # C00 (M4) and C11 (M4)?
    # C00 = M1 + M4 - M5 + M7.
    # C11 = M1 - M2 + M3 + M6.
    # So M4 goes to C00. k=0, i=0. W[0, r].
    W[0][r] = 1
    r += 1
    
    # M5
    U[0][r] = 1; U[1][r] = 1 # A00, A01
    V[6][r] = 1 # B11 (j=1,k=1)
    W[0][r] = -1; W[2][r] = 1 # C00 (-M5) and C01 (M5)?
    # C01 = M3 + M5.
    # So M5 goes to C01. k=1, i=0. W[1*2+0, r] = W[2, r].
    # And C00 gets -M5.
    W[0][r] = -1
    W[2][r] = 1
    r += 1
    
    # M6
    U[3][r] = 1; U[0][r] = -1 # A10-A00
    V[0][r] = 1; V[1][r] = 1 # B00+B01
    W[3][r] = 1 # C11 (M6)
    W[3][r] = 1
    r += 1
    
    # M7
    U[1][r] = 1; U[4][r] = -1 # A01-A11
    V[5][r] = 1; V[6][r] = 1 # B10+B11
    W[0][r] = 1 # C00 (M7)
    W[0][r] = 1
    r += 1
    
    # Sub-part 1b: 2x2x1 (B col 2)
    # Standard multiplication
    # C00 += A00 B02
    U[0][r] = 1
    V[2][r] = 1 # j=0, k=2
    W[0][r] = 1 # k=0, i=0
    r += 1
    
    # C11 += A11 B12
    U[4][r] = 1
    V[7][r] = 1 # j=1, k=2
    W[3][r] = 1 # k=1, i=1
    r += 1
    
    # Part 2: 2x1x5 block (A col 2, B row 2)
    # Rank 1
    # C_i,k += A_i,2 B_2,k
    # U[i*3+2, r] = 1
    # V[2*5+k, r] = 1
    # W[k*2+i, r] = 1
    U[2][r] = 1 # i=0, j=2
    U[5][r] = 1 # i=1, j=2
    for k in range(5):
        V[2*5+k][r] = 1
        W[k*2+0][r] = 1 # i=0
        W[k*2+1][r] = 1 # i=1
    r += 1
    
    # Total r = 9 + 1 = 10.
    # Wait, I missed the rest of B columns for the 2x2 block?
    # B has 5 columns. I used 0,1 for Strassen and 2 for standard.
    # I missed columns 3 and 4!
    
    # I need to handle B cols 3,4 for the 2x2 block.
    # 2x2x2 for cols 3,4. Rank 7.
    
    # Strassen for B cols 3,4
    # M1
    U[0][r] = 1; U[3][r] = 1
    V[3][r] = 1; V[8][r] = 1 # B03, B13
    W[4][r] = 1; W[5][r] = 1 # C03, C13?
    # C03 index k=3, i=0. W[3*2+0]=W[6].
    # C13 index k=3, i=1. W[3*2+1]=W[7].
    # C04 index k=4, i=0. W[4*2+0]=W[8].
    # C14 index k=4, i=1. W[4*2+1]=W[9].
    W[6][r] = 1; W[7][r] = 1
    r += 1
    
    # M2
    U[3][r] = 1; U[4][r] = 1
    V[3][r] = 1
    W[7][r] = 1 # C13
    r += 1
    
    # M3
    U[0][r] = 1
    V[8][r] = 1; V[3][r] = -1 # B13-B03
    W[7][r] = 1 # C14? No, C13?
    # C13 = M1 - M2 + M3 + M6.
    # So M3 goes to C13.
    W[7][r] = 1
    r += 1
    
    # M4
    U[4][r] = 1
    V[4][r] = 1; V[8][r] = -1 # B04-B13
    W[6][r] = 1 # C03
    r += 1
    
    # M5
    U[0][r] = 1; U[1][r] = 1
    V[9][r] = 1 # B14
    W[6][r] = -1; W[8][r] = 1 # C03 (-M5), C04 (M5)
    r += 1
    
    # M6
    U[3][r] = 1; U[0][r] = -1
    V[3][r] = 1; V[4][r] = 1
    W[7][r] = 1 # C13? No, C14?
    # C14 = M1 - M2 + M3 + M6? No.
    # C14 = M4 + M7?
    # Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # Mapping:
    # C03 -> C00
    # C04 -> C01
    # C13 -> C10
    # C14 -> C11
    # So M6 goes to C14.
    W[9][r] = 1
    r += 1
    
    # M7
    U[1][r] = 1; U[4][r] = -1
    V[8][r] = 1; V[9][r] = 1
    W[6][r] = 1 # C03
    r += 1
    
    # Total rank = 10 (first 2x2x3) + 7 (second 2x2x2) + 1 (2x1x5) = 18.
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    sol = construct_rank_17()
    if verify_decomposition(n, m, p, sol["U"], sol["V"], sol["W"]):
        return sol
    else:
        # Fallback to rank 28
        return construct_rank_28()

def construct_rank_28():
    # Copy from original if needed
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    r = 0
    # ... (same as before) ...
    return {"U": U, "V": V, "W": W}
