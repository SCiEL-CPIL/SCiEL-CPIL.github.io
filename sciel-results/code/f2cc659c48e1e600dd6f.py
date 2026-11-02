# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 2, 2, 7
    rank = 25
    
    # Initialize U, V, W as numpy arrays for easier manipulation
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2x2
    # U (A) coeffs for M1..M7
    # A indices: 0:A00, 1:A01, 2:A10, 3:A11
    strassen_U = np.array([
        [1, 0, 0, 1], 
        [0, 0, 1, 1], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0],
        [0, 1, 0, -1] 
    ])
    
    # W (C) coeffs for M1..M7
    # Local C indices: 0:C00, 1:C01, 2:C10, 3:C11
    # Mapping to global W index: k*n + i
    # For block b, k is 2b or 2b+1.
    # We will construct the full W contribution for each block.
    
    # V (B) coeffs for M1..M7
    # Local B indices: 0:B00, 1:B01, 2:B10, 3:B11
    # Mapping to global V index: j*p + k
    # For block b, k is 2b or 2b+1.
    # j=0 -> k=2b, 2b+1. j=1 -> k=2b, 2b+1.
    
    col = 0
    
    # Process 3 Strassen blocks
    for b in range(3):
        for r in range(7):
            # U
            U[:, col] += strassen_U[r]
            
            # V
            # strassen_V_local[r] needs to be mapped to global V indices
            # Let's define the local V coefficients explicitly for speed
            # M1: B00, B11 -> V[2b], V[2b+1] (for j=0) and V[7+2b], V[7+2b+1] (for j=1)?
            # No, V index is j*p + k.
            # j=0: V[k]. j=1: V[7+k].
            # Local B00 is B[0, 2b] -> V[2b]
            # Local B01 is B[0, 2b+1] -> V[2b+1]
            # Local B10 is B[1, 2b] -> V[7+2b]
            # Local B11 is B[1, 2b+1] -> V[7+2b+1]
            
            # Coefficients for V for each Strassen step r
            # r=0 (M1): B00(1), B11(1) -> V[2b]+=1, V[7+2b+1]+=1
            # r=1 (M2): B00(1) -> V[2b]+=1
            # r=2 (M3): B01(1), B11(-1) -> V[2b+1]+=1, V[7+2b+1]+=-1
            # r=3 (M4): B10(1), B00(-1) -> V[7+2b]+=1, V[2b]+=-1
            # r=4 (M5): B11(1) -> V[7+2b+1]+=1
            # r=5 (M6): B00(1), B01(1) -> V[2b]+=1, V[2b+1]+=1
            # r=6 (M7): B10(1), B11(1) -> V[7+2b]+=1, V[7+2b+1]+=1
            
            if r == 0:
                V[2*b, col] += 1
                V[7+2*b+1, col] += 1
            elif r == 1:
                V[2*b, col] += 1
            elif r == 2:
                V[2*b+1, col] += 1
                V[7+2*b+1, col] += -1
            elif r == 3:
                V[7+2*b, col] += 1
                V[2*b, col] += -1
            elif r == 4:
                V[7+2*b+1, col] += 1
            elif r == 5:
                V[2*b, col] += 1
                V[2*b+1, col] += 1
            elif r == 6:
                V[7+2*b, col] += 1
                V[7+2*b+1, col] += 1
            
            # W
            # Local C indices: 0:C00, 1:C01, 2:C10, 3:C11
            # Global W index: k*n + i
            # k=2b: W[2b*2 + i] = W[4b + i]
            # k=2b+1: W[(2b+1)*2 + i] = W[4b + 2 + i]
            # C00: i=0, k=2b -> W[4b]
            # C01: i=1, k=2b -> W[4b+1]
            # C10: i=0, k=2b+1 -> W[4b+2]
            # C11: i=1, k=2b+1 -> W[4b+3]
            
            # strassen_W_local[r]
            # r=0 (M1): C00(1), C11(1) -> W[4b]+=1, W[4b+3]+=1
            # r=1 (M2): C10(1), C11(-1) -> W[4b+2]+=1, W[4b+3]+=-1
            # r=2 (M3): C01(1), C11(1) -> W[4b+1]+=1, W[4b+3]+=1
            # r=3 (M4): C00(1), C10(1) -> W[4b]+=1, W[4b+2]+=1
            # r=4 (M5): C00(-1), C01(1) -> W[4b]+=-1, W[4b+1]+=1
            # r=5 (M6): C11(1) -> W[4b+3]+=1
            # r=6 (M7): C00(1) -> W[4b]+=1
            
            if r == 0:
                W[4*b, col] += 1
                W[4*b+3, col] += 1
            elif r == 1:
                W[4*b+2, col] += 1
                W[4*b+3, col] += -1
            elif r == 2:
                W[4*b+1, col] += 1
                W[4*b+3, col] += 1
            elif r == 3:
                W[4*b, col] += 1
                W[4*b+2, col] += 1
            elif r == 4:
                W[4*b, col] += -1
                W[4*b+1, col] += 1
            elif r == 5:
                W[4*b+3, col] += 1
            elif r == 6:
                W[4*b, col] += 1
            
            col += 1
            
    # Schoolbook for last block (col 6)
    # S1: A00 B0 -> C0
    # S2: A01 B1 -> C0
    # S3: A10 B0 -> C1
    # S4: A11 B1 -> C1
    
    # U for schoolbook
    # S1: A00 -> U[0]+=1
    # S2: A01 -> U[1]+=1
    # S3: A10 -> U[2]+=1
    # S4: A11 -> U[3]+=1
    
    # V for schoolbook
    # S1: B0 (B[0,6]) -> V[6]+=1
    # S2: B1 (B[1,6]) -> V[7+6]+=1 -> V[13]+=1
    # S3: B0 (B[0,6]) -> V[6]+=1
    # S4: B1 (B[1,6]) -> V[13]+=1
    
    # W for schoolbook
    # S1: C0 (C[0,6]) -> W[6*2+0] = W[12]+=1
    # S2: C0 (C[0,6]) -> W[12]+=1
    # S3: C1 (C[1,6]) -> W[6*2+1] = W[13]+=1
    # S4: C1 (C[1,6]) -> W[13]+=1
    
    # r=0 (S1)
    U[0, col] += 1
    V[6, col] += 1
    W[12, col] += 1
    col += 1
    
    # r=1 (S2)
    U[1, col] += 1
    V[13, col] += 1
    W[12, col] += 1
    col += 1
    
    # r=2 (S3)
    U[2, col] += 1
    V[6, col] += 1
    W[13, col] += 1
    col += 1
    
    # r=3 (S4)
    U[3, col] += 1
    V[13, col] += 1
    W[13, col] += 1
    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
