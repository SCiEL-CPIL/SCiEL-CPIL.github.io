import time

DIMENSIONS = [2, 2, 16]

def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using Strassen for 2x2x2 blocks
    # 8 blocks of 2x2 in B
    num_blocks = p // 2
    rank_per_block = 7
    rank = num_blocks * rank_per_block
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B10)
    # P4 = A11(B10-B01)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    # Mapping for block b (cols 2b, 2b+1)
    # A indices: 0:00, 1:01, 2:10, 3:11
    # B indices for block b:
    # B00: 0*p + 2b
    # B01: 0*p + 2b+1
    # B10: 1*p + 2b
    # B11: 1*p + 2b+1
    
    # W indices for block b:
    # C00 (i=0, k=2b): 2b*n + 0
    # C10 (i=1, k=2b): 2b*n + 1
    # C01 (i=0, k=2b+1): (2b+1)*n + 0
    # C11 (i=1, k=2b+1): (2b+1)*n + 1
    
    for b in range(num_blocks):
        base_r = b * rank_per_block
        # B indices
        b00 = 0 * p + 2 * b
        b01 = 0 * p + 2 * b + 1
        b10 = 1 * p + 2 * b
        b11 = 1 * p + 2 * b + 1
        
        # W indices
        w00 = 2 * b * n + 0
        w10 = 2 * b * n + 1
        w01 = (2 * b + 1) * n + 0
        w11 = (2 * b + 1) * n + 1
        
        # P1
        r = base_r + 0
        U[0][r] = 1
        U[3][r] = 1
        V[b00][r] = 1
        V[b11][r] = 1
        W[w00][r] = 1
        W[w11][r] = 1
        
        # P2
        r = base_r + 1
        U[2][r] = 1
        U[3][r] = 1
        V[b00][r] = 1
        W[w10][r] = 1
        
        # P3
        r = base_r + 2
        U[0][r] = 1
        V[b01][r] = 1
        V[b10][r] = -1
        W[w01][r] = 1
        W[w11][r] = -1
        
        # P4
        r = base_r + 3
        U[3][r] = 1
        V[b10][r] = 1
        V[b01][r] = -1
        W[w00][r] = -1
        W[w10][r] = 1
        
        # P5
        r = base_r + 4
        U[0][r] = 1
        U[1][r] = 1
        V[b11][r] = 1
        W[w01][r] = 1
        
        # P6
        r = base_r + 5
        U[2][r] = 1
        U[0][r] = -1
        V[b00][r] = 1
        V[b01][r] = 1
        W[w11][r] = 1
        
        # P7
        r = base_r + 6
        U[1][r] = 1
        U[3][r] = -1
        V[b10][r] = 1
        V[b11][r] = 1
        W[w00][r] = 1
        
    return {"U": U, "V": V, "W": W}
