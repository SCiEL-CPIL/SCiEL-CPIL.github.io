"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 51
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # For each block of 3 columns in B (and C), we decompose:
    # Part 1: A[:, 0:2] (2x2) * B[0:2, k] (2x3) -> C[:, k] (2x3)
    #   For k=0,1: use Strassen (7 terms)
    #   For k=2: use trivial (4 terms)
    # Part 2: A[:, 2] (2x1) * B[2, k] (1x3) -> C[:, k] (2x3)
    #   For each k: 2 terms (outer product of 2x1 and 1x1)
    # Total per block: 7 + 4 + 6 = 17 terms
    # Total: 3 blocks * 17 = 51 terms
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1
    
    for block in range(3):
        # Part 1: Strassen for A[:, 0:2] * B[0:2, block*3: block*3+2]
        # This is 2x2 * 2x2 -> 2x2, using 7 terms
        # A indices: i=0,1; j=0,1
        # B indices: j=0,1; k=block*3, block*3+1
        # C indices: i=0,1; k=block*3, block*3+1
        
        # M1 = A[0,0] * B[0,0]
        add_term(0, 1, 0, 1, block*3*2 + 0, 1)  # U[0], V[0], W[block*3*2+0]
        
        # M2 = A[1,0] * B[0,0]
        add_term(1, 1, 0, 1, block*3*2 + 1, 1)  # U[1], V[0], W[block*3*2+1]
        
        # M3 = A[0,1] * B[1,0]
        add_term(2, 1, 3, 1, block*3*2 + 0, 1)  # U[2], V[3], W[block*3*2+0]
        
        # M4 = A[1,1] * B[1,0]
        add_term(3, 1, 3, 1, block*3*2 + 1, 1)  # U[3], V[3], W[block*3*2+1]
        
        # M5 = (A[0,0]+A[0,1]) * (B[1,0]+B[1,1])
        add_term(0, 1, 3, 1, block*3*2 + 0, 1)  # U[0], V[3], W[block*3*2+0]
        add_term(2, 1, 3, 1, block*3*2 + 0, 1)  # U[2], V[3], W[block*3*2+0]
        add_term(0, 1, 4, 1, block*3*2 + 1, 1)  # U[0], V[4], W[block*3*2+1]
        add_term(2, 1, 4, 1, block*3*2 + 1, 1)  # U[2], V[4], W[block*3*2+1]
        
        # Wait, this is getting complicated. Let me use a cleaner Strassen decomposition.
        # Actually, let me just use the standard 7-term Strassen for 2x2*2x2.
        
        # Let me restart with a cleaner approach.
        pass
    
    # Let me use a different, cleaner approach.
    # I'll construct the matrices by explicitly listing all 51 terms.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    for block in range(3):
        # Part 1: Strassen for 2x2 * 2x2 (A[:,0:2] * B[0:2, block*3:block*3+2])
        # 7 terms
        
        # Term 1: M1 = A[0,0] * B[0,0]
        U[0][col] = 1
        V[0][col] = 1
        W[(block*3)*2 + 0][col] = 1
        col += 1
        
        # Term 2: M2 = A[1,0] * B[0,0]
        U[1][col] = 1
        V[0][col] = 1
        W[(block*3)*2 + 1][col] = 1
        col += 1
        
        # Term 3: M3 = A[0,1] * B[1,0]
        U[2][col] = 1
        V[3][col] = 1
        W[(block*3)*2 + 0][col] = 1
        col += 1
        
        # Term 4: M4 = A[1,1] * B[1,0]
        U[3][col] = 1
        V[3][col] = 1
        W[(block*3)*2 + 1][col] = 1
        col += 1
        
        # Term 5: M5 = (A[0,0]+A[0,1]) * (B[1,0]+B[1,1])
        # This contributes to C[0,1] with +1 and C[1,1] with +1
        # Actually, let me use the standard Strassen formulas:
        # M1 = A00*B00
        # M2 = A10*B00
        # M3 = A01*B10
        # M4 = A11*B10
        # M5 = (A00+A01)*(B10+B11)
        # M6 = (A10+A11)*B01
        # M7 = A00*(B11-B01)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let me redo this properly.
        
        # I'll just use the trivial decomposition for now to ensure correctness,
        # then optimize.
        pass
    
    # Let me just use the trivial decomposition with rank 54, which is known to be correct.
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
