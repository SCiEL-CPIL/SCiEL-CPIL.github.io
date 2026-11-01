"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # p = 12
    # We split p into 6 blocks of size 2
    num_blocks = p // 2
    block_size = 2
    
    # Strassen rank is 7 for 2x2x2
    rank_per_block = 7
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for b in range(num_blocks):
        # k indices for this block: b*2, b*2+1
        k0 = b * block_size
        k1 = b * block_size + 1
        
        # Strassen terms for 2x2 matrix multiplication
        # A indices: 00, 01, 10, 11 -> U index i*m+j
        # B indices: 00, 01, 10, 11 -> V index j*p+k
        # C indices: 00, 01, 10, 11 -> W index k*n+i
        
        # M1 = (A00+A11)(B00+B11)
        U[0][col] = 1; U[3][col] = 1
        V[0*p + k0][col] = 1; V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1; W[k1*n + 0][col] = 1
        W[k0*n + 1][col] = 1; W[k1*n + 1][col] = 1
        # Wait, W is for C.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's define the 7 terms properly.
        # Term 1: M1
        # U: A00, A11
        # V: B00, B11
        # W: C00 (+1), C11 (+1) -- wait, M1 contributes to C00 and C11?
        # C00 = M1 + ...
        # C11 = M1 + ...
        # So W[k0, 0] += 1, W[k1, 1] += 1?
        # W index is k*n + i.
        # C00 is i=0, k=k0. W[k0*2 + 0].
        # C11 is i=1, k=k1. W[k1*2 + 1].
        
        # Let's restart the term definition.
        # M1 = (A00+A11)(B00+B11)
        # U: [1, 0, 0, 1]
        # V: [1, 0, 0, 1] (for B00, B11) -> V[0*p+k0], V[1*p+k1]
        # W: C00 gets +1, C11 gets +1.
        # W[k0*2+0] += 1
        # W[k1*2+1] += 1
        
        # M2 = (A10+A11)B00
        # U: [0, 0, 1, 1]
        # V: [1, 0, 0, 0] -> V[0*p+k0]
        # W: C10 gets +1, C11 gets -1.
        # W[k0*2+1] += 1
        # W[k1*2+1] += -1
        
        # M3 = A00(B01-B11)
        # U: [1, 0, 0, 0]
        # V: [0, 1, 0, -1] -> V[0*p+k1], V[1*p+k1]
        # W: C01 gets +1, C11 gets +1.
        # W[k1*2+0] += 1
        # W[k1*2+1] += 1
        
        # M4 = A11(B10-B00)
        # U: [0, 0, 0, 1]
        # V: [0, 0, 1, -1] -> V[1*p+k0], V[0*p+k0]
        # W: C00 gets +1, C10 gets +1.
        # W[k0*2+0] += 1
        # W[k0*2+1] += 1
        
        # M5 = (A00+A01)B11
        # U: [1, 1, 0, 0]
        # V: [0, 0, 0, 1] -> V[1*p+k1]
        # W: C00 gets -1, C01 gets +1.
        # W[k0*2+0] += -1
        # W[k1*2+0] += 1
        
        # M6 = (A10-A00)(B00+B01)
        # U: [0, 0, 1, -1]
        # V: [1, 1, 0, 0] -> V[0*p+k0], V[0*p+k1]
        # W: C11 gets +1.
        # W[k1*2+1] += 1
        
        # M7 = (A01-A11)(B10+B11)
        # U: [0, 1, 0, -1]
        # V: [0, 0, 1, 1] -> V[1*p+k0], V[1*p+k1]
        # W: C00 gets +1.
        # W[k0*2+0] += 1
        
        # Let's code this explicitly.
        
        # Term 1
        U[0][col] += 1; U[3][col] += 1
        V[0*p + k0][col] += 1; V[1*p + k1][col] += 1
        W[k0*n + 0][col] += 1
        W[k1*n + 1][col] += 1
        col += 1
        
        # Term 2
        U[2][col] += 1; U[3][col] += 1
        V[0*p + k0][col] += 1
        W[k0*n + 1][col] += 1
        W[k1*n + 1][col] += -1
        col += 1
        
        # Term 3
        U[0][col] += 1
        V[0*p + k1][col] += 1; V[1*p + k1][col] += -1
        W[k1*n + 0][col] += 1
        W[k1*n + 1][col] += 1
        col += 1
        
        # Term 4
        U[3][col] += 1
        V[1*p + k0][col] += 1; V[0*p + k0][col] += -1
        W[k0*n + 0][col] += 1
        W[k0*n + 1][col] += 1
        col += 1
        
        # Term 5
        U[0][col] += 1; U[1][col] += 1
        V[1*p + k1][col] += 1
        W[k0*n + 0][col] += -1
        W[k1*n + 0][col] += 1
        col += 1
        
        # Term 6
        U[2][col] += 1; U[0][col] += -1
        V[0*p + k0][col] += 1; V[0*p + k1][col] += 1
        W[k1*n + 1][col] += 1
        col += 1
        
        # Term 7
        U[1][col] += 1; U[3][col] += -1
        V[1*p + k0][col] += 1; V[1*p + k1][col] += 1
        W[k0*n + 0][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
