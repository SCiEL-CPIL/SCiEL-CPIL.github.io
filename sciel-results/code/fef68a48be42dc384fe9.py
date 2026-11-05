import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # Rank 55 decomposition
    # Split A (2x3) into A1 (2x2) and A2 (2x1)
    # Split B (3x10) into B1 (2x10) and B2 (1x10)
    # C = A1 B1 + A2 B2
    
    # Part 1: A1 (2x2) * B1 (2x10)
    # B1 is 2x10. Split into 5 blocks of 2x2.
    # Each 2x2x2 uses rank 7 (Strassen).
    # Total rank for this part: 5 * 7 = 35.
    
    # Part 2: A2 (2x1) * B2 (1x10)
    # B2 is 1x10. Split into 5 blocks of 1x2.
    # Each 2x1x2 uses rank 4 (schoolbook, as rank 3 is not sufficient for separate terms).
    # Total rank for this part: 5 * 4 = 20.
    
    # Total rank: 35 + 20 = 55.
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # --- Part 1: 2x2x10 using Strassen for 5 blocks of 2x2 ---
    # A1 indices: A[0,0], A[0,1], A[1,0], A[1,1] -> U indices 0,1,2,3
    # B1 indices: B[0,0..9], B[1,0..9] -> V indices 0..9, 10..19
    # C1 indices: C[0,0..9], C[1,0..9] -> W indices 0..9, 10..19 (since p=10, n=2, W[k*n+i])
    
    for block in range(5):
        # Block b corresponds to B columns 2*b, 2*b+1
        # B1 block is 2x2: B[0, 2b], B[0, 2b+1], B[1, 2b], B[1, 2b+1]
        # V indices for this block:
        # v0 = 0 * 10 + 2*b = 2*b
        # v1 = 0 * 10 + 2*b + 1 = 2*b + 1
        # v2 = 1 * 10 + 2*b = 10 + 2*b
        # v3 = 1 * 10 + 2*b + 1 = 10 + 2*b + 1
        
        # Strassen 2x2x2 decomposition
        # m1 = (A00+A11)(B00+B11)
        # m2 = (A10+A11)B00
        # m3 = A00(B10-B11)
        # m4 = A11(B00-B10)
        # m5 = (A00+A01)B11
        # m6 = (A10-A00)(B00+B10)
        # m7 = (A01-A11)(B10+B11)
        
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        # U indices for A1:
        # u0 = 0 (A00)
        # u1 = 1 (A01)
        # u2 = 2 (A10)
        # u3 = 3 (A11)
        
        # W indices for C1 block:
        # C1 is 2x2.
        # C[0, 2b] -> W[2b * 2 + 0] = 4b
        # C[0, 2b+1] -> W[(2b+1) * 2 + 0] = 4b + 2
        # C[1, 2b] -> W[2b * 2 + 1] = 4b + 1
        # C[1, 2b+1] -> W[(2b+1) * 2 + 1] = 4b + 3
        
        w00 = 4 * block
        w01 = 4 * block + 2
        w10 = 4 * block + 1
        w11 = 4 * block + 3
        
        # Term 1: m1 = (A00+A11)(B00+B11)
        U[0][col] += 1; U[3][col] += 1
        V[2*block][col] += 1; V[10+2*block+1][col] += 1
        W[w00][col] += 1; W[w11][col] += 1
        col += 1
        
        # Term 2: m2 = (A10+A11)B00
        U[2][col] += 1; U[3][col] += 1
        V[2*block][col] += 1
        W[w10][col] += 1
        col += 1
        
        # Term 3: m3 = A00(B10-B11)
        U[0][col] += 1
        V[10+2*block][col] += 1; V[10+2*block+1][col] -= 1
        W[w01][col] += 1
        col += 1
        
        # Term 4: m4 = A11(B00-B10)
        U[3][col] += 1
        V[2*block][col] += 1; V[10+2*block][col] -= 1
        W[w00][col] += 1; W[w10][col] += 1
        col += 1
        
        # Term 5: m5 = (A00+A01)B11
        U[0][col] += 1; U[1][col] += 1
        V[10+2*block+1][col] += 1
        W[w00][col] -= 1; W[w01][col] += 1
        col += 1
        
        # Term 6: m6 = (A10-A00)(B00+B10)
        U[2][col] += 1; U[0][col] -= 1
        V[2*block][col] += 1; V[10+2*block][col] += 1
        W[w11][col] += 1
        col += 1
        
        # Term 7: m7 = (A01-A11)(B10+B11)
        U[1][col] += 1; U[3][col] -= 1
        V[10+2*block][col] += 1; V[10+2*block+1][col] += 1
        W[w00][col] += 1; W[w11][col] += 1
        col += 1

    # --- Part 2: 2x1x10 using schoolbook for 5 blocks of 1x2 ---
    # A2 indices: A[0,2], A[1,2] -> U indices 4, 5
    # B2 indices: B[2,0..9] -> V indices 20..29
    # C2 indices: C[0,0..9], C[1,0..9] -> W indices 0..9, 10..19
    
    for block in range(5):
        # Block b corresponds to B2 columns 2*b, 2*b+1
        # B2 block is 1x2: B[2, 2b], B[2, 2b+1]
        # V indices:
        # v0 = 2 * 10 + 2*b = 20 + 2*b
        # v1 = 2 * 10 + 2*b + 1 = 20 + 2*b + 1
        
        # U indices:
        # u0 = 4 (A02)
        # u1 = 5 (A12)
        
        # W indices for C2 block:
        # C[0, 2b] -> W[4b]
        # C[0, 2b+1] -> W[4b+2]
        # C[1, 2b] -> W[4b+1]
        # C[1, 2b+1] -> W[4b+3]
        
        w00 = 4 * block
        w01 = 4 * block + 2
        w10 = 4 * block + 1
        w11 = 4 * block + 3
        
        # Term 1: A02 * B20
        U[4][col] += 1
        V[20+2*block][col] += 1
        W[w00][col] += 1
        col += 1
        
        # Term 2: A02 * B21
        U[4][col] += 1
        V[20+2*block+1][col] += 1
        W[w01][col] += 1
        col += 1
        
        # Term 3: A12 * B20
        U[5][col] += 1
        V[20+2*block][col] += 1
        W[w10][col] += 1
        col += 1
        
        # Term 4: A12 * B21
        U[5][col] += 1
        V[20+2*block+1][col] += 1
        W[w11][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
