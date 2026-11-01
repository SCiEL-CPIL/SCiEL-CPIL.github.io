"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 5 blocks of 2x2x2 Strassen (rank 7)
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # Each entry is (u_idx, val), (v_idx, val), (w_idx, val)
    # u_idx in 0..3 (i*m+j)
    # v_idx in 0..19 (j*p+k)
    # w_idx in 0..19 (k*n+i)
    
    # We define the 7 terms for a block with columns k0, k1
    # Term 1: (a00+a11)(b0k0+b1k1)
    # U: (0, 1), (3, 1)
    # V: (10*k0, 1), (10*k1+1, 1)
    # W: (2*k0+0, 1), (2*k1+1, 1)
    
    # Term 2: (a10+a11)b0k0
    # U: (2, 1), (3, 1)
    # V: (10*k0, 1)
    # W: (2*k0+1, 1), (2*k1+1, -1)
    
    # Term 3: a00(b0k1-b1k1)
    # U: (0, 1)
    # V: (10*k1, 1), (10*k1+1, -1)
    # W: (2*k1+0, 1), (2*k1+1, 1)
    
    # Term 4: a11(b1k0-b0k0)
    # U: (3, 1)
    # V: (10*k0+1, 1), (10*k0, -1)
    # W: (2*k0+0, 1), (2*k0+1, 1)
    
    # Term 5: (a00+a10)b1k1
    # U: (0, 1), (2, 1)
    # V: (10*k1+1, 1)
    # W: (2*k0+0, -1), (2*k1+0, 1)
    
    # Term 6: (a01-a00)(b0k0+b1k0)
    # U: (1, 1), (0, -1)
    # V: (10*k0, 1), (10*k0+1, 1)
    # W: (2*k1+1, 1)
    
    # Term 7: (a11-a10)(b0k1+b1k1)
    # U: (3, 1), (2, -1)
    # V: (10*k1, 1), (10*k1+1, 1)
    # W: (2*k0+0, 1)
    
    col = 0
    for j in range(5):
        k0 = 2 * j
        k1 = 2 * j + 1
        
        # Term 1
        U[0][col] += 1
        U[3][col] += 1
        V[10 * k0][col] += 1
        V[10 * k1 + 1][col] += 1
        W[2 * k0 + 0][col] += 1
        W[2 * k1 + 1][col] += 1
        col += 1
        
        # Term 2
        U[2][col] += 1
        U[3][col] += 1
        V[10 * k0][col] += 1
        W[2 * k0 + 1][col] += 1
        W[2 * k1 + 1][col] += -1
        col += 1
        
        # Term 3
        U[0][col] += 1
        V[10 * k1][col] += 1
        V[10 * k1 + 1][col] += -1
        W[2 * k1 + 0][col] += 1
        W[2 * k1 + 1][col] += 1
        col += 1
        
        # Term 4
        U[3][col] += 1
        V[10 * k0 + 1][col] += 1
        V[10 * k0][col] += -1
        W[2 * k0 + 0][col] += 1
        W[2 * k0 + 1][col] += 1
        col += 1
        
        # Term 5
        U[0][col] += 1
        U[2][col] += 1
        V[10 * k1 + 1][col] += 1
        W[2 * k0 + 0][col] += -1
        W[2 * k1 + 0][col] += 1
        col += 1
        
        # Term 6
        U[1][col] += 1
        U[0][col] += -1
        V[10 * k0][col] += 1
        V[10 * k0 + 1][col] += 1
        W[2 * k1 + 1][col] += 1
        col += 1
        
        # Term 7
        U[3][col] += 1
        U[2][col] += -1
        V[10 * k1][col] += 1
        V[10 * k1 + 1][col] += 1
        W[2 * k0 + 0][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
