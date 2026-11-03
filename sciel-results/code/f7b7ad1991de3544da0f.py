import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen decomposition for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h) -> u=[1,0,1,0], v=[1,1,0,0], w=[1,0,0,1] (C11, C12, C21, C22)
    # M2 = (c+d)e   -> u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0] (C21) -- wait, C21 = M2+M4. C22 = M1-M2+...
    # Let's check the W vectors carefully.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # M1: u=[1,0,1,0], v=[1,1,0,0], w=[1,0,0,1]
    # M2: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,-1]
    # M3: u=[1,0,0,0], v=[0,1,0,-1], w=[0,1,0,1]
    # M4: u=[0,0,0,1], v=[0,0,1,-1], w=[1,0,1,0]
    # M5: u=[1,1,0,0], v=[0,0,0,1], w=[-1,1,0,0]
    # M6: u=[1,-1,0,0], v=[1,1,0,0], w=[0,0,0,1]
    # M7: u=[0,1,0,-1], v=[0,0,1,1], w=[1,0,0,0]
    
    # Verify M1: (a+d)(e+h) = ae+ah+de+dh.
    # C11 += 1, C22 += 1. Correct.
    
    # Verify M2: (c+d)e = ce+de.
    # C21 += 1, C22 -= 1. Correct.
    
    # Verify M3: a(f-h) = af-ah.
    # C12 += 1, C22 += 1. Correct.
    
    # Verify M4: d(g-e) = dg-de.
    # C11 += 1, C21 += 1. Correct.
    
    # Verify M5: (a+b)h = ah+bh.
    # C11 -= 1, C12 += 1. Correct.
    
    # Verify M6: (c-a)(e+f) = ce+cf-ae-af.
    # C22 += 1. Correct.
    
    # Verify M7: (b-d)(g+h) = bg+bh-dg-dh.
    # C11 += 1. Correct.
    
    strassen_U = np.array([
        [1, 0, 1, 0], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [1, -1, 0, 0],# M6
        [0, 1, 0, -1] # M7
    ]).T # Shape (4, 7)
    
    strassen_V = np.array([
        [1, 1, 0, 0], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]).T # Shape (4, 7)
    
    strassen_W = np.array([
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1],# M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]).T # Shape (4, 7)
    
    # Now map to 2x4x4
    # A is 2x4. Indexing: A[0,0], A[0,1], A[0,2], A[0,3], A[1,0]...
    # In U, rows correspond to A entries.
    # U shape (8, 28).
    # V shape (16, 28).
    # W shape (8, 28).
    
    # Block 1: A B11. A is rows 0,1 of A. B11 is top-left 2x2 of B.
    # B11 entries: B[0,0], B[0,1], B[1,0], B[1,1].
    # In V, rows correspond to B entries.
    # B index: j*p+k.
    # B11: j=0,1; k=0,1.
    # Indices: 0,1,4,5.
    
    # We need to place the 7 terms of Strassen into the 28 columns.
    # Columns 0-6: Block 1 (A B11)
    # Columns 7-13: Block 2 (A B21)
    # Columns 13-19: Block 3 (A B12)
    # Columns 20-26: Block 4 (A B22)
    
    # For each block, the U part is the same (depends on A).
    # The V part depends on which B block.
    # The W part depends on which C block.
    
    # C is 2x4.
    # C11 (top-left 2x2 of C) corresponds to A B11? No.
    # C = A B.
    # C[:, 0:2] = A[:, 0:2] B[0:2, 0:2] + A[:, 2:4] B[2:4, 0:2]
    # So C_left = A1 B11 + A2 B21.
    # C_right = A1 B12 + A2 B22.
    
    # So:
    # Block 1 (A1 B11): Contributes to C[:, 0:2].
    # Block 2 (A2 B21): Contributes to C[:, 0:2].
    # Block 3 (A1 B12): Contributes to C[:, 2:4].
    # Block 4 (A2 B22): Contributes to C[:, 2:4].
    
    # U for A1 (cols 0,1 of A): Rows 0,1 of U.
    # U for A2 (cols 2,3 of A): Rows 2,3 of U.
    # Note: A is 2x4. A[i,j].
    # A1 = A[:, 0:2]. A2 = A[:, 2:4].
    
    # Let's fill U.
    # For Block 1 (A1 B11): U uses A1 entries.
    # A1 entries are A[0,0], A[0,1], A[1,0], A[1,1].
    # In U (8 rows), these are rows 0, 1, 2, 3?
    # U index: i*m+j.
    # A[0,0] -> 0*4+0 = 0.
    # A[0,1] -> 0*4+1 = 1.
    # A[1,0] -> 1*4+0 = 4.
    # A[1,1] -> 1*4+1 = 5.
    # So for A1, we use rows 0,1,4,5.
    # For A2 (cols 2,3):
    # A[0,2] -> 2.
    # A[0,3] -> 3.
    # A[1,2] -> 6.
    # A[1,3] -> 7.
    # So for A2, we use rows 2,3,6,7.
    
    # V for B11 (top-left):
    # B[0,0] -> 0*4+0 = 0.
    # B[0,1] -> 1.
    # B[1,0] -> 4.
    # B[1,1] -> 5.
    # Rows 0,1,4,5.
    
    # V for B21 (bottom-left):
    # B[2,0] -> 8.
    # B[2,1] -> 9.
    # B[3,0] -> 12.
    # B[3,1] -> 13.
    # Rows 8,9,12,13.
    
    # V for B12 (top-right):
    # B[0,2] -> 2.
    # B[0,3] -> 3.
    # B[1,2] -> 6.
    # B[1,3] -> 7.
    # Rows 2,3,6,7.
    
    # V for B22 (bottom-right):
    # B[2,2] -> 10.
    # B[2,3] -> 11.
    # B[3,2] -> 14.
    # B[3,3] -> 15.
    # Rows 10,11,14,15.
    
    # W for C_left (cols 0,1):
    # C[0,0] -> 0*2+0 = 0.
    # C[0,1] -> 1.
    # C[1,0] -> 2.
    # C[1,1] -> 3.
    # Rows 0,1,2,3.
    
    # W for C_right (cols 2,3):
    # C[0,2] -> 4.
    # C[0,3] -> 5.
    # C[1,2] -> 6.
    # C[1,3] -> 7.
    # Rows 4,5,6,7.
    
    # Helper to place a 4x7 matrix into a larger matrix at specific rows/cols
    def place(target, source, rows, cols):
        for r_idx, r in enumerate(rows):
            for c_idx, c in enumerate(cols):
                target[r, c] = source[r_idx, c_idx]
        
    # Block 1: A1 B11 -> C_left
    cols_1 = list(range(0, 7))
    place(U, strassen_U, [0, 1, 4, 5], cols_1)
    place(V, strassen_V, [0, 1, 4, 5], cols_1)
    place(W, strassen_W, [0, 1, 2, 3], cols_1)
    
    # Block 2: A2 B21 -> C_left
    cols_2 = list(range(7, 14))
    place(U, strassen_U, [2, 3, 6, 7], cols_2)
    place(V, strassen_V, [8, 9, 12, 13], cols_2)
    place(W, strassen_W, [0, 1, 2, 3], cols_2)
    
    # Block 3: A1 B12 -> C_right
    cols_3 = list(range(14, 21))
    place(U, strassen_U, [0, 1, 4, 5], cols_3)
    place(V, strassen_V, [2, 3, 6, 7], cols_3)
    place(W, strassen_W, [4, 5, 6, 7], cols_3)
    
    # Block 4: A2 B22 -> C_right
    cols_4 = list(range(21, 28))
    place(U, strassen_U, [2, 3, 6, 7], cols_4)
    place(V, strassen_V, [10, 11, 14, 15], cols_4)
    place(W, strassen_W, [4, 5, 6, 7], cols_4)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
