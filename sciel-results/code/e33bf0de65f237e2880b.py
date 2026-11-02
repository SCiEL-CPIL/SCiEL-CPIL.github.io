"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Process 5 blocks of 2 columns each for Strassen on A1 (2x2) * B1 (2x2)
    # A1 uses A cols 0, 1. B1 uses B rows 0, 1.
    # B cols are grouped into 5 pairs: (0,1), (2,3), (4,5), (6,7), (8,9)
    
    for block_idx in range(5):
        k0 = block_idx * 2
        k1 = block_idx * 2 + 1
        
        # Local indices for A1 (2x2)
        # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
        idx_a = 0 * 3 + 0
        idx_b = 0 * 3 + 1
        idx_c = 1 * 3 + 0
        idx_d = 1 * 3 + 1
        
        # Local indices for B1 (2x2)
        # e=B[0,k0], f=B[0,k1], g=B[1,k0], h=B[1,k1]
        idx_e = 0 * 10 + k0
        idx_f = 0 * 10 + k1
        idx_g = 1 * 10 + k0
        idx_h = 1 * 10 + k1
        
        # W indices
        # C[0,k0] -> W[k0*2+0]
        # C[0,k1] -> W[k1*2+0]
        # C[1,k0] -> W[k0*2+1]
        # C[1,k1] -> W[k1*2+1]
        w_00 = k0 * 2 + 0
        w_01 = k1 * 2 + 0
        w_10 = k0 * 2 + 1
        w_11 = k1 * 2 + 1
        
        # M1 = (a+d)(e+g)
        U[idx_a][col] += 1
        U[idx_d][col] += 1
        V[idx_e][col] += 1
        V[idx_g][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        col += 1
        
        # M2 = (c+d)e
        U[idx_c][col] += 1
        U[idx_d][col] += 1
        V[idx_e][col] += 1
        W[w_10][col] += 1
        col += 1
        
        # M3 = a(f-h)
        U[idx_a][col] += 1
        V[idx_f][col] += 1
        V[idx_h][col] -= 1
        W[w_01][col] += 1
        W[w_11][col] += 1
        col += 1
        
        # M4 = d(g-e)
        U[idx_d][col] += 1
        V[idx_g][col] += 1
        V[idx_e][col] -= 1
        W[w_00][col] += 1
        W[w_10][col] += 1
        col += 1
        
        # M5 = (a+b)h
        U[idx_a][col] += 1
        U[idx_b][col] += 1
        V[idx_h][col] += 1
        W[w_00][col] -= 1
        W[w_01][col] += 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[idx_c][col] += 1
        U[idx_a][col] -= 1
        V[idx_e][col] += 1
        V[idx_f][col] += 1
        W[w_11][col] += 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[idx_b][col] += 1
        U[idx_d][col] -= 1
        V[idx_g][col] += 1
        V[idx_h][col] += 1
        W[w_00][col] += 1
        col += 1

    # Process A2 (2x1) * B2 (1x10)
    # A2 uses A col 2. B2 uses B row 2.
    idx_a2_0 = 0 * 3 + 2
    idx_a2_1 = 1 * 3 + 2
    
    for k in range(10):
        idx_b2_k = 2 * 10 + k
        
        # T1 = A[0,2] * B[2,k]
        U[idx_a2_0][col] += 1
        V[idx_b2_k][col] += 1
        W[k * 2 + 0][col] += 1
        col += 1
        
        # T2 = A[1,2] * B[2,k]
        U[idx_a2_1][col] += 1
        V[idx_b2_k][col] += 1
        W[k * 2 + 1][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
