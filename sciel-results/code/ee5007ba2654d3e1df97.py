import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 13

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    term = 0

    # --- Term 2: A_R B_R (2x1x3) ---
    # A_R: A[0,2] (idx 2), A[1,2] (idx 5)
    # B_R: B[2,0] (idx 6), B[2,1] (idx 7), B[2,2] (idx 8)
    # C_R: C[0,0] (idx 0), C[0,1] (idx 2), C[0,2] (idx 4)
    #      C[1,0] (idx 1), C[1,1] (idx 3), C[1,2] (idx 5)
    
    # T1: A[0,2] * B[2,:] * C[0,:]
    U[2, term] = 1
    V[6, term] = 1; V[7, term] = 1; V[8, term] = 1
    W[0, term] = 1; W[2, term] = 1; W[4, term] = 1
    term += 1

    # T2: A[1,2] * B[2,:] * C[1,:]
    U[5, term] = 1
    V[6, term] = 1; V[7, term] = 1; V[8, term] = 1
    W[1, term] = 1; W[3, term] = 1; W[5, term] = 1
    term += 1

    # --- Subterm 1b: A_L B_L2 (2x2x1) ---
    # A_L: A[0,0] (0), A[0,1] (1), A[1,0] (3), A[1,1] (4)
    # B_L2: B[0,2] (2), B[1,2] (5)
    # C_L2: C[0,2] (4), C[1,2] (5)
    
    # T3: A[0,0] * B[0,2] * C[0,2]
    U[0, term] = 1
    V[2, term] = 1
    W[4, term] = 1
    term += 1

    # T4: A[1,0] * B[0,2] * C[1,2]
    U[3, term] = 1
    V[2, term] = 1
    W[5, term] = 1
    term += 1

    # T5: A[0,1] * B[1,2] * C[0,2]
    U[1, term] = 1
    V[5, term] = 1
    W[4, term] = 1
    term += 1

    # T6: A[1,1] * B[1,2] * C[1,2]
    U[4, term] = 1
    V[5, term] = 1
    W[5, term] = 1
    term += 1

    # --- Subterm 1a: A_L B_L1 (2x2x2) Strassen ---
    # A_L: A00(0), A01(1), A10(3), A11(4)
    # B_L1: B00(0), B01(1), B10(3), B11(4)
    # C_L1: C00(0), C01(2), C10(1), C11(3)
    
    # M1 = (A00+A11)(B00+B11)
    U[0, term] = 1; U[4, term] = 1
    V[0, term] = 1; V[4, term] = 1
    W[0, term] = 1; W[3, term] = 1
    term += 1

    # M2 = (A10+A11)B00
    U[3, term] = 1; U[4, term] = 1
    V[0, term] = 1
    W[1, term] = 1; W[3, term] = -1
    term += 1

    # M3 = A00(B01-B11)
    U[0, term] = 1
    V[1, term] = 1; V[4, term] = -1
    W[2, term] = 1; W[3, term] = 1
    term += 1

    # M4 = A11(B10-B00)
    U[4, term] = 1
    V[3, term] = 1; V[0, term] = -1
    W[0, term] = 1; W[1, term] = 1
    term += 1

    # M5 = (A00+A01)B11
    U[0, term] = 1; U[1, term] = 1
    V[4, term] = 1
    W[0, term] = -1; W[2, term] = 1
    term += 1

    # M6 = (A10-A00)(B00+B01)
    U[3, term] = 1; U[0, term] = -1
    V[0, term] = 1; V[1, term] = 1
    W[3, term] = 1
    term += 1

    # M7 = (A01-A11)(B10+B11)
    U[1, term] = 1; U[4, term] = -1
    V[3, term] = 1; V[4, term] = 1
    W[0, term] = 1
    term += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
