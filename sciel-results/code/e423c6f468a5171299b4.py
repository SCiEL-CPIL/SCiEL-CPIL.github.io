"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use 6 blocks of 2x2 Strassen
    # Rank is 6 * 7 = 42
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+g)
    # M2 = (c+d)(e+g)
    # M3 = a(f-g)
    # M4 = d(h-e)
    # M5 = (a+b)(h)
    # M6 = (c+b)(f)
    # M7 = (a-d)(e)
    
    # Note: The standard Strassen decomposition maps to the tensor.
    # We need to map the indices correctly.
    # U has shape (n*m, rank) = (4, 42)
    # V has shape (m*p, rank) = (24, 42)
    # W has shape (p*n, rank) = (24, 42)
    
    # Indices for A (U): i*m + j. i in 0..1, j in 0..1.
    # 0: a (0,0), 1: b (0,1), 2: c (1,0), 3: d (1,1)
    
    # Indices for B (V): j*p + k. j in 0..1, k in 0..11.
    # Block b (0..5) uses columns 2*b, 2*b+1.
    # For block b, B_block = [[B[0, 2b], B[0, 2b+1]], [B[1, 2b], B[1, 2b+1]]]
    # Let e = B[0, 2b], f = B[0, 2b+1], g = B[1, 2b], h = B[1, 2b+1]
    # V indices for e: 0*p + 2b = 2b
    # V indices for f: 0*p + 2b+1 = 2b+1
    # V indices for g: 1*p + 2b = 12 + 2b
    # V indices for h: 1*p + 2b+1 = 12 + 2b+1
    
    # Indices for C (W): k*n + i. k in 0..11, i in 0..1.
    # C[0, 2b] is from block b, col 0. Index: (2b)*2 + 0 = 4b
    # C[0, 2b+1] is from block b, col 1. Index: (2b+1)*2 + 0 = 4b+2
    # C[1, 2b] is from block b, col 0. Index: (2b)*2 + 1 = 4b+1
    # C[1, 2b+1] is from block b, col 1. Index: (2b+1)*2 + 1 = 4b+3
    
    # Strassen steps for block b:
    # col_base = 2*b
    # v_e = 2*col_base
    # v_f = 2*col_base + 1
    # v_g = 12 + 2*col_base
    # v_h = 12 + 2*col_base + 1
    
    # w_c11 = 4*col_base
    # w_c12 = 4*col_base + 2
    # w_c21 = 4*col_base + 1
    # w_c22 = 4*col_base + 3
    
    # u_a = 0, u_b = 1, u_c = 2, u_d = 3
    
    for b in range(6):
        offset = b * 7
        col_base = 2 * b
        
        v_e = 2 * col_base
        v_f = 2 * col_base + 1
        v_g = 12 + 2 * col_base
        v_h = 12 + 2 * col_base + 1
        
        w_c11 = 4 * col_base
        w_c12 = 4 * col_base + 2
        w_c21 = 4 * col_base + 1
        w_c22 = 4 * col_base + 3
        
        # M1 = (a+d)(e+g)
        U[0][offset] = 1
        U[3][offset] = 1
        V[v_e][offset] = 1
        V[v_g][offset] = 1
        W[w_c11][offset] = 1
        W[w_c22][offset] = 1
        W[w_c21][offset] = -1
        W[w_c12][offset] = -1
        
        # M2 = (c+d)(e+g)
        U[2][offset+1] = 1
        U[3][offset+1] = 1
        V[v_e][offset+1] = 1
        V[v_g][offset+1] = 1
        W[w_c21][offset+1] = 1
        W[w_c22][offset+1] = -1
        
        # M3 = a(f-g)
        U[0][offset+2] = 1
        V[v_f][offset+2] = 1
        V[v_g][offset+2] = -1
        W[w_c12][offset+2] = 1
        W[w_c22][offset+2] = 1
        
        # M4 = d(h-e)
        U[3][offset+3] = 1
        V[v_h][offset+3] = 1
        V[v_e][offset+3] = -1
        W[w_c11][offset+3] = 1
        W[w_c21][offset+3] = 1
        
        # M5 = (a+b)(h)
        U[0][offset+4] = 1
        U[1][offset+4] = 1
        V[v_h][offset+4] = 1
        W[w_c12][offset+4] = 1
        W[w_c22][offset+4] = 1
        
        # M6 = (c+b)(f)
        U[2][offset+5] = 1
        U[1][offset+5] = 1
        V[v_f][offset+5] = 1
        W[w_c11][offset+5] = 1
        W[w_c21][offset+5] = 1
        
        # M7 = (a-d)(e)
        U[0][offset+6] = 1
        U[3][offset+6] = -1
        V[v_e][offset+6] = 1
        W[w_c11][offset+6] = 1
        W[w_c22][offset+6] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
