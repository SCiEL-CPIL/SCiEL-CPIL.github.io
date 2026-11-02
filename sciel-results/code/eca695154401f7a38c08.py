"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 11
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Term 0: A00 * B00 -> C00
    U[0][0] = 1
    V[0][0] = 1
    W[0][0] = 1
    
    # Term 1: A00 * B01 -> C01
    U[0][1] = 1
    V[1][1] = 1
    W[1][1] = 1
    
    # Term 2: A00 * B02 -> C02
    U[0][2] = 1
    V[2][2] = 1
    W[2][2] = 1
    
    # Term 3: A01 * B00 -> C10
    U[1][3] = 1
    V[0][3] = 1
    W[1][3] = 1 # W index k*n+i = 0*2+1 = 1? No. W[k*n+i]. k=0, i=1 -> idx 1.
    # Wait, W index for C10 is k=0, i=1. Index = 0*2 + 1 = 1.
    # My previous manual check: W[10] means row 1, col 0? No, W is (p*n, rank).
    # W[row, col]. Row = k*n + i.
    # C10: k=0, i=1. Row = 1.
    # C11: k=1, i=1. Row = 3.
    # C12: k=2, i=1. Row = 5.
    # C00: k=0, i=0. Row = 0.
    # C01: k=1, i=0. Row = 2.
    # C02: k=2, i=0. Row = 4.
    
    # Let's re-verify indices carefully.
    # U[i*m + j]
    # V[j*p + k]
    # W[k*n + i]
    
    # Term 0: A00 (i=0,j=0) * B00 (j=0,k=0) -> C00 (k=0,i=0)
    U[0][0] = 1
    V[0][0] = 1
    W[0][0] = 1
    
    # Term 1: A00 (i=0,j=0) * B01 (j=0,k=1) -> C01 (k=1,i=0)
    U[0][1] = 1
    V[1][1] = 1
    W[2][1] = 1 # k=1, i=0 -> 1*2+0 = 2
    
    # Term 2: A00 (i=0,j=0) * B02 (j=0,k=2) -> C02 (k=2,i=0)
    U[0][2] = 1
    V[2][2] = 1
    W[4][2] = 1 # k=2, i=0 -> 2*2+0 = 4
    
    # Term 3: A01 (i=0,j=1) * B00 (j=0,k=0) -> C10 (k=0,i=1)
    U[1][3] = 1
    V[0][3] = 1
    W[1][3] = 1 # k=0, i=1 -> 0*2+1 = 1
    
    # Term 4: A01 (i=0,j=1) * B01 (j=0,k=1) -> C11 (k=1,i=1)
    U[1][4] = 1
    V[1][4] = 1
    W[3][4] = 1 # k=1, i=1 -> 1*2+1 = 3
    
    # Term 5: A01 (i=0,j=1) * B02 (j=0,k=2) -> C12 (k=2,i=1)
    U[1][5] = 1
    V[2][5] = 1
    W[5][5] = 1 # k=2, i=1 -> 2*2+1 = 5
    
    # Term 6: A10 (i=1,j=0) * B10 (j=1,k=0) -> C00 (k=0,i=0)
    U[2][6] = 1
    V[3][6] = 1 # j=1, k=0 -> 1*3+0 = 3
    W[0][6] = 1
    
    # Term 7: A10 (i=1,j=0) * B11 (j=1,k=1) -> C01 (k=1,i=0)
    U[2][7] = 1
    V[4][7] = 1 # j=1, k=1 -> 1*3+1 = 4
    W[2][7] = 1
    
    # Term 8: A10 (i=1,j=0) * B12 (j=1,k=2) -> C02 (k=2,i=0)
    U[2][8] = 1
    V[5][8] = 1 # j=1, k=2 -> 1*3+2 = 5
    W[4][8] = 1
    
    # Term 9: A11 (i=1,j=1) * (B10+B11) -> C10+C11
    # U11 = 1
    # V10 = 1, V11 = 1
    # W10 = 1, W11 = 1
    U[3][9] = 1
    V[3][9] = 1 # j=1, k=0
    V[4][9] = 1 # j=1, k=1
    W[1][9] = 1 # k=0, i=1
    W[3][9] = 1 # k=1, i=1
    
    # Term 10: A11 (i=1,j=1) * B12 -> C12
    U[3][10] = 1
    V[5][10] = 1 # j=1, k=2
    W[5][10] = 1 # k=2, i=1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
