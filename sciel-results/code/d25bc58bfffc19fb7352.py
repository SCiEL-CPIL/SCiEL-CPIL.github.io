"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    rank = 18
    
    # Initialize matrices
    # U: (n*m, rank) = (4, 18)
    # V: (m*p, rank) = (10, 18)
    # W: (p*n, rank) = (10, 18)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, r):
        U[u_idx][r] += u_val
        V[v_idx][r] += v_val
        W[w_idx][r] += w_val

    # Part 1: Strassen for B cols 0, 1
    # B indices: j*5 + k. j=0,1. k=0,1.
    # V indices: 0, 1, 5, 6
    # W indices: k*2 + i. k=0,1. i=0,1.
    # W indices: 0, 1, 2, 3
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    add_term(0, 1, 0, 1, 0, 1, 0)
    add_term(3, 1, 6, 1, 0, 1, 0) # Wait, A11 is idx 3. B11 is idx 6. C00 is W[0], C11 is W[3].
    # Correction: add_term adds to specific indices.
    # M1:
    U[0][0] += 1
    U[3][0] += 1
    V[0][0] += 1
    V[6][0] += 1
    W[0][0] += 1 # C00
    W[3][0] += 1 # C11
    
    # M2: (A10+A11)B00 -> C10
    U[2][1] += 1
    U[3][1] += 1
    V[0][1] += 1
    W[1][1] += 1 # C10
    
    # M3: A00(B01-B11) -> C01
    U[0][2] += 1
    V[1][2] += 1
    V[6][2] += -1
    W[1][2] += 1 # C01
    
    # M4: A11(B10-B00) -> C00
    U[3][3] += 1
    V[5][3] += 1
    V[0][3] += -1
    W[0][3] += 1 # C00
    
    # M5: (A00+A01)B11 -> C00(-1), C01(1)
    U[0][4] += 1
    U[1][4] += 1
    V[6][4] += 1
    W[0][4] += -1 # C00
    W[1][4] += 1  # C01
    
    # M6: (A10-A00)(B00+B01) -> C11
    U[2][5] += 1
    U[0][5] += -1
    V[0][5] += 1
    V[1][5] += 1
    W[3][5] += 1  # C11
    
    # M7: (A01-A11)(B10+B11) -> C10(1), C00(-1)
    U[1][6] += 1
    U[3][6] += -1
    V[5][6] += 1
    V[6][6] += 1
    W[1][6] += 1  # C10
    W[0][6] += -1 # C00

    # Part 2: Strassen for B cols 2, 3
    # V indices: j*5+k. j=0,1. k=2,3.
    # V[2], V[3], V[7], V[8]
    # W indices: k*2+i. k=2,3. i=0,1.
    # W[4], W[5], W[6], W[7]
    
    # M'1: (A00+A11)(B02+B13) -> C02, C13
    U[0][7] += 1
    U[3][7] += 1
    V[2][7] += 1
    V[8][7] += 1
    W[4][7] += 1 # C02
    W[7][7] += 1 # C13
    
    # M'2: (A10+A11)B02 -> C12
    U[2][8] += 1
    U[3][8] += 1
    V[2][8] += 1
    W[5][8] += 1 # C12
    
    # M'3: A00(B03-B13) -> C03
    U[0][9] += 1
    V[3][9] += 1
    V[8][9] += -1
    W[6][9] += 1 # C03
    
    # M'4: A11(B12-B02) -> C02
    U[3][10] += 1
    V[7][10] += 1
    V[2][10] += -1
    W[4][10] += 1 # C02
    
    # M'5: (A00+A01)B13 -> C02(-1), C03(1)
    U[0][11] += 1
    U[1][11] += 1
    V[8][11] += 1
    W[4][11] += -1 # C02
    W[6][11] += 1  # C03
    
    # M'6: (A10-A00)(B02+B03) -> C13
    U[2][12] += 1
    U[0][12] += -1
    V[2][12] += 1
    V[3][12] += 1
    W[7][12] += 1  # C13
    
    # M'7: (A01-A11)(B12+B13) -> C12(1), C02(-1)
    U[1][13] += 1
    U[3][13] += -1
    V[7][13] += 1
    V[8][13] += 1
    W[5][13] += 1  # C12
    W[4][13] += -1 # C02

    # Part 3: Schoolbook for B col 4
    # V indices: j*5+4. V[4], V[9]
    # W indices: k*2+i. k=4. W[8], W[9]
    
    # M''1: A00 B04 -> C04
    U[0][14] += 1
    V[4][14] += 1
    W[8][14] += 1
    
    # M''2: A01 B14 -> C04
    U[1][15] += 1
    V[9][15] += 1
    W[8][15] += 1
    
    # M''3: A10 B04 -> C14
    U[2][16] += 1
    V[4][16] += 1
    W[9][16] += 1
    
    # M''4: A11 B14 -> C14
    U[3][17] += 1
    V[9][17] += 1
    W[9][17] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
