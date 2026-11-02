import numpy as np

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose into M_2,2,2 (rank 7) and M_2,2,3 (rank 12)
    rank = 19
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: Strassen for k=0, 1
    # Indices for A: i*m+j (0..3)
    # Indices for B: j*p+k (0..9)
    # Indices for C: k*n+i (0..9)
    
    # Strassen multipliers for 2x2x2
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    col = 0
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff_u, coeff_v, coeff_w):
        nonlocal col
        U[u_idx][col] += coeff_u
        V[v_idx][col] += coeff_v
        W[w_idx][col] += coeff_w
        col += 1

    # Note: The decomposition requires U, V, W to be linear.
    # Strassen uses sums/differences.
    # U entries can be 1, -1, 0.
    # V entries can be 1, -1, 0.
    # W entries can be 1, -1, 0.
    
    # M1: (A00 + A11) * (B00 + B11)
    # A indices: 00 (0), 11 (3)
    # B indices: 00 (0), 11 (3) -- wait, B index is j*p+k.
    # j=0,k=0 -> 0. j=1,k=1 -> 1*2+1 = 3.
    # C indices: C00 (0*2+0=0), C11 (1*2+1=3)
    
    # U for M1: A00 + A11
    U[0][col] = 1
    U[3][col] = 1
    # V for M1: B00 + B11
    V[0][col] = 1
    V[3][col] = 1
    # W for M1: C00 + C11
    W[0][col] = 1
    W[3][col] = 1
    col += 1
    
    # M2: (A10 + A11) * B00
    # A10 (2), A11 (3). B00 (0). C10 (1*2+0=2)
    U[2][col] = 1
    U[3][col] = 1
    V[0][col] = 1
    W[2][col] = 1
    col += 1
    
    # M3: A00 * (B01 - B11)
    # A00 (0). B01 (1), B11 (3). C01 (1*2+0? No. C index k*n+i. k=1, i=0 -> 1*2+0=2? No.
    # C index: k*n+i.
    # k=1, i=0 -> 2.
    # k=1, i=1 -> 3.
    # C01 is i=0, k=1. Index 1*2+0 = 2.
    # C11 is i=1, k=1. Index 1*2+1 = 3.
    # Wait, C01 is row 0, col 1.
    # My W index mapping: W[k*n+i].
    # For C01: k=1, i=0. Index 2.
    # For C11: k=1, i=1. Index 3.
    
    # Let's re-verify C indices.
    # C is 2x5.
    # C[i,k].
    # W index is k*n+i.
    # C00: k=0, i=0 -> 0.
    # C01: k=1, i=0 -> 2.
    # C10: k=0, i=1 -> 1.
    # C11: k=1, i=1 -> 3.
    
    # So:
    # M1 contributes to C00 (0) and C11 (3).
    # M2 contributes to C10 (1).
    # M3 contributes to C01 (2).
    # M4 contributes to C00 (0) and C10 (1).
    # M5 contributes to C01 (2) and C00 (0)?
    # Let's check Strassen formulas again.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M3: A00 * (B01 - B11)
    # A00 (0). B01 (1), B11 (3).
    # W index for C01 is 2.
    U[0][col] = 1
    V[1][col] = 1
    V[3][col] = -1
    W[2][col] = 1
    col += 1
    
    # M4: A11 * (B10 - B00)
    # A11 (3). B10 (2), B00 (0).
    # W indices: C00 (0), C10 (1).
    U[3][col] = 1
    V[2][col] = 1
    V[0][col] = -1
    W[0][col] = 1
    W[1][col] = 1
    col += 1
    
    # M5: (A00 + A01) * B11
    # A00 (0), A01 (1). B11 (3).
    # W indices: C00 (0) is subtracted? No, M5 is in C00 with -1, in C01 with +1.
    # Wait, W matrix holds the coefficients for C.
    # If W[0][col] = -1 and W[2][col] = 1, it means -M5 + M5?
    # The decomposition is sum_r U V W.
    # So W entries are the coefficients.
    U[0][col] = 1
    U[1][col] = 1
    V[3][col] = 1
    W[0][col] = -1
    W[2][col] = 1
    col += 1
    
    # M6: (A10 - A00) * (B00 + B01)
    # A10 (2), A00 (0). B00 (0), B01 (1).
    # W index: C11 (3).
    U[2][col] = 1
    U[0][col] = -1
    V[0][col] = 1
    V[1][col] = 1
    W[3][col] = 1
    col += 1
    
    # M7: (A01 - A11) * (B10 + B11)
    # A01 (1), A11 (3). B10 (2), B11 (3).
    # W indices: C00 (0).
    U[1][col] = 1
    U[3][col] = -1
    V[2][col] = 1
    V[3][col] = 1
    W[0][col] = 1
    col += 1
    
    # Part 2: Schoolbook for k=2,3,4
    # i in 0..1, j in 0..1, k in 2..4
    for i in range(n):
        for j in range(m):
            for k in range(2, p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
