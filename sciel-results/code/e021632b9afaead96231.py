import time
import numpy as np

DIMENSIONS = [2, 3, 10]
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 2.0

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-55 exact tensor decomposition for 2x3 @ 3x10.
    
    The decomposition groups the 10 columns of the second matrix into 5 pairs.
    For each pair, it performs a 2x3 @ 3x2 multiplication.
    The 2x3 @ 3x2 multiplication is split into:
    1. A 2x2 @ 2x2 multiplication using Strassen's algorithm (rank 7).
    2. A 2x1 @ 1x2 multiplication using schoolbook (rank 4).
    Total rank per block = 11.
    Total rank = 5 blocks * 11 = 55.
    """
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    
    # U shape: (n*m, rank) = (6, 55)
    # V shape: (m*p, rank) = (30, 55)
    # W shape: (p*n, rank) = (20, 55)
    
    rank = 55
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Indices for U (A matrix 2x3)
    # i in 0..1, j in 0..2
    # idx = i * 3 + j
    # A = [[U[0], U[1], U[2]],
    #      [U[3], U[4], U[5]]]
    
    # Indices for V (B matrix 3x10)
    # j in 0..2, k in 0..9
    # idx = j * 10 + k
    # B = [[V[0], V[1], ..., V[9]],
    #      [V[10], V[11], ..., V[19]],
    #      [V[20], V[21], ..., V[29]]]
    
    # Indices for W (C matrix 2x10)
    # k in 0..9, i in 0..1
    # idx = k * 2 + i
    # C = [[W[0], W[2], W[4], ..., W[18]],
    #      [W[1], W[3], W[5], ..., W[19]]]
    # Note: W[k*n+i] means W[k*2+i].
    
    # We process 5 blocks of 2 columns each.
    for k_block in range(5):
        # Columns in B: 2*k_block, 2*k_block + 1
        c1 = 2 * k_block
        c2 = 2 * k_block + 1
        
        # V indices for the 2x2 block (j=0,1)
        v_00 = 0 * 10 + c1
        v_01 = 0 * 10 + c2
        v_10 = 1 * 10 + c1
        v_11 = 1 * 10 + c2
        
        # V indices for the 1x2 block (j=2)
        v_20 = 2 * 10 + c1
        v_21 = 2 * 10 + c2
        
        # W indices for the 2x2 output block
        # C[i, c] -> idx = c * 2 + i
        w_00 = c1 * 2 + 0
        w_01 = c2 * 2 + 0
        w_10 = c1 * 2 + 1
        w_11 = c2 * 2 + 1
        
        base_col = k_block * 11
        
        # Strassen 2x2x2 (7 terms)
        # A_11 = U[0], A_12 = U[1], A_21 = U[3], A_22 = U[4]
        
        # P1 = (A11 + A22) * (B11 + B22)
        col = base_col
        U[0][col] += 1.0
        U[4][col] += 1.0
        V[v_00][col] += 1.0
        V[v_11][col] += 1.0
        W[w_00][col] += 1.0
        W[w_11][col] += 1.0
        
        # P2 = (A21 + A22) * B11
        col = base_col + 1
        U[3][col] += 1.0
        U[4][col] += 1.0
        V[v_00][col] += 1.0
        W[w_10][col] += 1.0
        W[w_11][col] -= 1.0
        
        # P3 = A11 * (B22 - B12)
        col = base_col + 2
        U[0][col] += 1.0
        V[v_11][col] += 1.0
        V[v_01][col] -= 1.0
        W[w_01][col] += 1.0
        W[w_11][col] += 1.0
        
        # P4 = A22 * (B21 - B11)
        col = base_col + 3
        U[4][col] += 1.0
        V[v_10][col] += 1.0
        V[v_00][col] -= 1.0
        W[w_00][col] += 1.0
        W[w_10][col] += 1.0
        
        # P5 = (A11 + A12) * B22
        col = base_col + 4
        U[0][col] += 1.0
        U[1][col] += 1.0
        V[v_11][col] += 1.0
        W[w_00][col] -= 1.0
        W[w_01][col] += 1.0
        
        # P6 = (A21 - A11) * (B11 + B12)
        col = base_col + 5
        U[3][col] += 1.0
        U[0][col] -= 1.0
        V[v_00][col] += 1.0
        V[v_01][col] += 1.0
        W[w_11][col] += 1.0
        
        # P7 = (A12 - A22) * (B21 + B22)
        col = base_col + 6
        U[1][col] += 1.0
        U[4][col] -= 1.0
        V[v_10][col] += 1.0
        V[v_11][col] += 1.0
        W[w_00][col] += 1.0
        
        # Schoolbook 2x1 @ 1x2 (4 terms)
        # A_col = [U[2], U[5]]
        # B_row = [V[v_20], V[v_21]]
        
        # Q1 = A1 * B1
        col = base_col + 7
        U[2][col] += 1.0
        V[v_20][col] += 1.0
        W[w_00][col] += 1.0
        
        # Q2 = A1 * B2
        col = base_col + 8
        U[2][col] += 1.0
        V[v_21][col] += 1.0
        W[w_01][col] += 1.0
        
        # Q3 = A2 * B1
        col = base_col + 9
        U[5][col] += 1.0
        V[v_20][col] += 1.0
        W[w_10][col] += 1.0
        
        # Q4 = A2 * B2
        col = base_col + 10
        U[5][col] += 1.0
        V[v_21][col] += 1.0
        W[w_11][col] += 1.0

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
