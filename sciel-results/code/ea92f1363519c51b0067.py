"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Rank 33 construction using Strassen for 2x2 blocks and outer product for 2x1 blocks
    rank = 33
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 33
    V = [[0] * rank for _ in range(m * p)]  # 18 x 33
    W = [[0] * rank for _ in range(p * n)]  # 12 x 33
    
    # Indices for A (2x3): i in 0..1, j in 0..2 -> idx = i*3 + j
    # Indices for B (3x6): j in 0..2, k in 0..5 -> idx = j*6 + k
    # Indices for C (2x6): i in 0..1, k in 0..5 -> idx in W = k*2 + i
    
    for b in range(3):
        # Block b corresponds to columns 2b and 2b+1 of B and C
        
        # Strassen part for A[0:2, 0:2] * B[0:2, 2b:2b+2]
        # A indices
        a00 = 0 * 3 + 0
        a01 = 0 * 3 + 1
        a10 = 1 * 3 + 0
        a11 = 1 * 3 + 1
        
        # B indices (top 2 rows)
        b00 = 0 * 6 + 2 * b
        b01 = 0 * 6 + 2 * b + 1
        b10 = 1 * 6 + 2 * b
        b11 = 1 * 6 + 2 * b + 1
        
        # C indices (W indices)
        c00 = (2 * b) * 2 + 0
        c01 = (2 * b + 1) * 2 + 0
        c10 = (2 * b) * 2 + 1
        c11 = (2 * b + 1) * 2 + 1
        
        # M1
        col = 7 * b + 0
        U[a00][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        
        # M2
        col = 7 * b + 1
        U[a10][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1
        W[c11][col] -= 1
        
        # M3
        col = 7 * b + 2
        U[a00][col] += 1
        V[b10][col] += 1
        V[b11][col] -= 1
        W[c01][col] += 1
        W[c11][col] += 1
        
        # M4
        col = 7 * b + 3
        U[a11][col] += 1
        V[b00][col] += 1
        V[b01][col] -= 1
        W[c00][col] -= 1
        W[c10][col] += 1
        
        # M5
        col = 7 * b + 4
        U[a00][col] += 1
        U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        
        # M6
        col = 7 * b + 5
        U[a10][col] += 1
        U[a00][col] -= 1
        V[b00][col] += 1
        V[b01][col] += 1
        W[c11][col] += 1
        
        # M7
        col = 7 * b + 6
        U[a01][col] += 1
        U[a11][col] -= 1
        V[b10][col] += 1
        V[b11][col] += 1
        W[c01][col] += 1
        W[c00][col] -= 1
        
        # Outer product part for A[0:2, 2] * B[2, 2b:2b+2]
        # A indices
        ar0 = 0 * 3 + 2
        ar1 = 1 * 3 + 2
        
        # B indices (bottom row)
        br0 = 2 * 6 + 2 * b
        br1 = 2 * 6 + 2 * b + 1
        
        # N0: ar0 * br0 -> c00
        col = 21 + 4 * b + 0
        U[ar0][col] += 1
        V[br0][col] += 1
        W[c00][col] += 1
        
        # N1: ar0 * br1 -> c01
        col = 21 + 4 * b + 1
        U[ar0][col] += 1
        V[br1][col] += 1
        W[c01][col] += 1
        
        # N2: ar1 * br0 -> c10
        col = 21 + 4 * b + 2
        U[ar1][col] += 1
        V[br0][col] += 1
        W[c10][col] += 1
        
        # N3: ar1 * br1 -> c11
        col = 21 + 4 * b + 3
        U[ar1][col] += 1
        V[br1][col] += 1
        W[c11][col] += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
