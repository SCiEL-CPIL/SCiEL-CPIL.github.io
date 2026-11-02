import numpy as np
import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 120s limit, leave 10s margin
    
    n, m, p = DIMENSIONS
    
    # Construct one 3x2x2 block (Rank 11)
    # U_b: 6x11, V_b: 4x11, W_b: 6x11
    U_b = np.zeros((6, 11))
    V_b = np.zeros((4, 11))
    W_b = np.zeros((6, 11))
    
    # Strassen part (cols 0-6)
    # U indices: i*2 + j. i in 0,1. j in 0,1.
    # V indices: j*2 + k. j in 0,1. k in 0,1.
    # W indices: k*3 + i. k in 0,1. i in 0,2 (only 0,1 used).
    
    # s=0
    U_b[0, 0] = 1; U_b[3, 0] = 1 # A00, A11
    V_b[0, 0] = 1; V_b[3, 0] = 1 # B00, B11
    W_b[0, 0] = 1 # C00
    W_b[3, 0] = 1 # C10? No.
    # Let's re-verify W mapping.
    # W[k*n+i, c]. n=3.
    # C00: k=0, i=0 -> idx 0.
    # C01: k=0, i=1 -> idx 1.
    # C10: k=1, i=0 -> idx 3.
    # C11: k=1, i=1 -> idx 4.
    
    # Strassen Formulas:
    # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2: (A10+A11)B00 -> C10 += M2, C11 -= M2
    # M3: A00(B01-B11) -> C01 += M3, C11 += M3
    # M4: A11(B10-B00) -> C00 += M4, C10 += M4
    # M5: (A00+A01)B11 -> C00 -= M5, C01 += M5
    # M6: (A10-A00)(B00+B01) -> C11 += M6
    # M7: (A01-A11)(B10+B11) -> C00 += M7
    
    # Col 0 (M1)
    U_b[0, 0] = 1 # A00
    U_b[3, 0] = 1 # A11
    V_b[0, 0] = 1 # B00
    V_b[3, 0] = 1 # B11
    W_b[0, 0] = 1 # C00
    W_b[4, 0] = 1 # C11
    
    # Col 1 (M2)
    U_b[2, 1] = 1 # A10
    U_b[3, 1] = 1 # A11
    V_b[0, 1] = 1 # B00
    W_b[3, 1] = 1 # C10
    W_b[4, 1] = -1 # C11
    
    # Col 2 (M3)
    U_b[0, 2] = 1 # A00
    V_b[1, 2] = 1 # B01
    V_b[3, 2] = -1 # B11
    W_b[1, 2] = 1 # C01
    W_b[4, 2] = 1 # C11
    
    # Col 3 (M4)
    U_b[3, 3] = 1 # A11
    V_b[2, 3] = 1 # B10
    V_b[0, 3] = -1 # B00
    W_b[0, 3] = 1 # C00
    W_b[3, 3] = 1 # C10
    
    # Col 4 (M5)
    U_b[0, 4] = 1 # A00
    U_b[1, 4] = 1 # A01
    V_b[3, 4] = 1 # B11
    W_b[0, 4] = -1 # C00
    W_b[1, 4] = 1 # C01
    
    # Col 5 (M6)
    U_b[2, 5] = 1 # A10
    U_b[0, 5] = -1 # A00
    V_b[0, 5] = 1 # B00
    V_b[1, 5] = 1 # B01
    W_b[4, 5] = 1 # C11
    
    # Col 6 (M7)
    U_b[1, 6] = 1 # A01
    U_b[3, 6] = -1 # A11
    V_b[2, 6] = 1 # B10
    V_b[3, 6] = 1 # B11
    W_b[0, 6] = 1 # C00
    
    # Schoolbook part (cols 7-10)
    # i=2. A indices 4,5.
    # Col 7: A20 B00 -> C20
    U_b[4, 7] = 1
    V_b[0, 7] = 1
    W_b[2, 7] = 1 # k=0, i=2 -> idx 2
    
    # Col 8: A20 B01 -> C21
    U_b[4, 8] = 1
    V_b[1, 8] = 1
    W_b[5, 8] = 1 # k=1, i=2 -> idx 5
    
    # Col 9: A21 B10 -> C20
    U_b[5, 9] = 1
    V_b[2, 9] = 1
    W_b[2, 9] = 1
    
    # Col 10: A21 B11 -> C21
    U_b[5, 10] = 1
    V_b[3, 10] = 1
    W_b[5, 10] = 1
    
    # Global Matrices
    U = np.zeros((n*m, 44))
    V = np.zeros((m*p, 44))
    W = np.zeros((p*n, 44))
    
    # Block 0: A_L (j 0,1), B_TL (j 0,1; k 0,1)
    # U rows: i*4 + j. i 0..2, j 0..1.
    # V rows: j*4 + k. j 0..1, k 0..1.
    # W rows: k*3 + i. k 0..1, i 0..2.
    
    for c in range(11):
        # U
        for i in range(3):
            for j in range(2):
                val = U_b[i*2 + j, c]
                if val != 0:
                    U[i*4 + j, c] = val
        # V
        for j in range(2):
            for k in range(2):
                val = V_b[j*2 + k, c]
                if val != 0:
                    V[j*4 + k, c] = val
        # W
        for k in range(2):
            for i in range(3):
                val = W_b[k*3 + i, c]
                if val != 0:
                    W[k*3 + i, c] = val
                    
    # Block 1: A_L (j 0,1), B_TR (j 0,1; k 2,3)
    # Cols 11-21
    for c in range(11):
        gc = 11 + c
        # U is same as Block 0
        for i in range(3):
            for j in range(2):
                val = U_b[i*2 + j, c]
                if val != 0:
                    U[i*4 + j, gc] = val
        # V: j 0..1, k 2..3
        for j in range(2):
            for k in range(2):
                val = V_b[j*2 + k, c]
                if val != 0:
                    V[j*4 + (k+2), gc] = val
        # W: k 2..3, i 0..2
        for k in range(2):
            for i in range(3):
                val = W_b[k*3 + i, c]
                if val != 0:
                    W[(k+2)*3 + i, gc] = val

    # Block 2: A_R (j 2,3), B_BL (j 2,3; k 0,1)
    # Cols 22-32
    for c in range(11):
        gc = 22 + c
        # U: j 2..3
        for i in range(3):
            for j in range(2):
                val = U_b[i*2 + j, c]
                if val != 0:
                    U[i*4 + (j+2), gc] = val
        # V: j 2..3, k 0..1
        for j in range(2):
            for k in range(2):
                val = V_b[j*2 + k, c]
                if val != 0:
                    V[(j+2)*4 + k, gc] = val
        # W: k 0..1, i 0..2
        for k in range(2):
            for i in range(3):
                val = W_b[k*3 + i, c]
                if val != 0:
                    W[k*3 + i, gc] = val

    # Block 3: A_R (j 2,3), B_BR (j 2,3; k 2,3)
    # Cols 33-43
    for c in range(11):
        gc = 33 + c
        # U: j 2..3
        for i in range(3):
            for j in range(2):
                val = U_b[i*2 + j, c]
                if val != 0:
                    U[i*4 + (j+2), gc] = val
        # V: j 2..3, k 2..3
        for j in range(2):
            for k in range(2):
                val = V_b[j*2 + k, c]
                if val != 0:
                    V[(j+2)*4 + (k+2), gc] = val
        # W: k 2..3, i 0..2
        for k in range(2):
            for i in range(3):
                val = W_b[k*3 + i, c]
                if val != 0:
                    W[(k+2)*3 + i, gc] = val

    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
