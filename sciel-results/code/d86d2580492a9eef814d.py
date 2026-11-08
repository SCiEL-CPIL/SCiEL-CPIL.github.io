"""Rewritten program for matrix multiplication 3,4,8 using rank-88 decomposition."""
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 8
    rank = 88
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Part 1: i=0, Schoolbook for 1x4x8
    # C[0, k] = sum_j A[0, j] B[j, k]
    for j in range(4):
        for k in range(8):
            U[0 * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + 0, col] = 1
            col += 1
            
    # Part 2: i=1,2, Strassen for 2x4x8
    # Decompose 2x4x8 into 4 blocks of 2x2x2? No, 2x4x8 = 2 * (2x2x4) + 2 * (2x2x4)?
    # We split j into [0,1] and [2,3].
    # We split k into [0,1], [2,3], [4,5], [6,7].
    # For each j-block and k-block, we have a 2x2x2 multiplication?
    # A_sub is 2x2, B_sub is 2x2?
    # B is 4x8.
    # If we fix j-block [0,1] and k-block [0,1], B_sub is 2x2.
    # Then A_sub (2x2) * B_sub (2x2) -> C_sub (2x2).
    # This is a 2x2x2 multiplication.
    # There are 2 j-blocks and 4 k-blocks. Total 8 blocks?
    # Wait, the current code does 2 j-blocks * 4 k-blocks * 7 (Strassen) = 56.
    # This corresponds to 8 independent 2x2x2 Strassen decompositions.
    # Rank 8 * 7 = 56.
    
    # Strassen factors for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B00-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U factors (A side)
    # Indices: (li, lj) where li in {0,1}, lj in {0,1}
    # Flat index: li*2 + lj
    sU = np.array([
        [1, 0, 0, 1],  # M1: A00+A11
        [0, 0, 1, 1],  # M2: A10+A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00+A01
        [-1, 0, 1, 0], # M6: A10-A00
        [0, 1, 0, -1]  # M7: A01-A11
    ])
    
    # V factors (B side)
    # Indices: (lj, lk) where lj in {0,1}, lk in {0,1}
    sV = np.array([
        [1, 0, 0, 1],  # M1: B00+B11
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B10-B11
        [-1, 0, 1, 0], # M4: B00-B10
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00+B01
        [0, 0, 1, 1]   # M7: B10+B11
    ])
    
    # W factors (C side)
    # Indices: (li, lk) where li in {0,1}, lk in {0,1}
    sW = np.array([
        [1, 0, 0, 1],  # M1: C00+=1, C11+=1
        [0, 0, 1, -1], # M2: C10+=1, C11-=1
        [0, 1, 0, 1],  # M3: C01+=1, C11+=1
        [1, 0, 1, 0],  # M4: C00+=1, C10+=1
        [-1, 1, 0, 0], # M5: C00-=1, C01+=1
        [0, 0, 0, 1],  # M6: C11+=1
        [1, 0, 0, 0]   # M7: C00+=1
    ])
    
    for jb in range(2):  # j block index
        j0 = jb * 2
        for kb in range(4):  # k block index
            k0 = kb * 2
            for t in range(7):
                # Update U
                for li in range(2):
                    for lj in range(2):
                        l_idx = li * 2 + lj
                        val = sU[t, l_idx]
                        if val != 0:
                            U[(1 + li) * m + (j0 + lj), col] += val
                            
                # Update V
                for lj in range(2):
                    for lk in range(2):
                        l_idx = lj * 2 + lk
                        val = sV[t, l_idx]
                        if val != 0:
                            V[(j0 + lj) * p + (k0 + lk), col] += val
                            
                # Update W
                for li in range(2):
                    for lk in range(2):
                        l_idx = li * 2 + lk
                        val = sW[t, l_idx]
                        if val != 0:
                            W[(k0 + lk) * n + (1 + li), col] += val
                            
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
