"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We treat the 3x8 product as two independent 3x4 blocks.
    # Each 3x4 block is decomposed into a 2x2x4 part (rank 14) and a 2x1x4 part (rank 8).
    # Total rank per block = 22.
    # Total rank = 2 * 22 = 44.
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Function to add a 2x2x4 Strassen-like block (rank 14)
    # A indices: 0,1 (rows), 0,1 (cols)
    # B indices: 0,1 (rows), k0..k0+3 (cols)
    # C indices: 0,1 (rows), k0..k0+3 (cols)
    def add_2x2x4_block(k_start):
        nonlocal col
        # This part computes C[0:2, k_start:k_start+4] += A[0:2, 0:2] * B[0:2, k_start:k_start+4]
        # We split B into two 2x2 blocks: B1 (cols k_start, k_start+1) and B2 (cols k_start+2, k_start+3)
        # And use Strassen for each? No, R(2,2,4)=14 is better than 7+7=14.
        # Actually, R(2,2,4) can be done with 14 mults.
        # One way: 2 Strassen blocks (14) is the standard way.
        # Let's use two Strassen blocks of size 2x2x2.
        
        # Block 1: C[0:2, k_start:k_start+2] from A[0:2, 0:2] * B[0:2, k_start:k_start+2]
        k0 = k_start
        k1 = k_start + 1
        
        # M1 = (A00 + A11) * (B0k0 + B1k1)
        U[0 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M2 = (A10 + A11) * B0k0
        U[1 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1
        
        # M3 = A00 * (B0k1 - B1k1)
        U[0 * m + 0][col] = 1
        V[0 * p + k1][col] = 1
        V[1 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M4 = A11 * (B1k0 - B0k0)
        U[1 * m + 1][col] = 1
        V[1 * p + k0][col] = 1
        V[0 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # M5 = (A00 + A01) * B1k1
        U[0 * m + 0][col] = 1
        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # M6 = (A10 - A00) * (B0k0 + B0k1)
        U[1 * m + 0][col] = 1
        U[0 * m + 0][col] = -1
        V[0 * p + k0][col] = 1
        V[0 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M7 = (A01 - A11) * (B1k0 + B1k1)
        U[0 * m + 1][col] = 1
        U[1 * m + 1][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1

        # Block 2: C[0:2, k_start+2:k_start+4] from A[0:2, 0:2] * B[0:2, k_start+2:k_start+4]
        k0 = k_start + 2
        k1 = k_start + 3
        
        # M1 = (A00 + A11) * (B0k0 + B1k1)
        U[0 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M2 = (A10 + A11) * B0k0
        U[1 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1
        
        # M3 = A00 * (B0k1 - B1k1)
        U[0 * m + 0][col] = 1
        V[0 * p + k1][col] = 1
        V[1 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M4 = A11 * (B1k0 - B0k0)
        U[1 * m + 1][col] = 1
        V[1 * p + k0][col] = 1
        V[0 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # M5 = (A00 + A01) * B1k1
        U[0 * m + 0][col] = 1
        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # M6 = (A10 - A00) * (B0k0 + B0k1)
        U[1 * m + 0][col] = 1
        U[0 * m + 0][col] = -1
        V[0 * p + k0][col] = 1
        V[0 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M7 = (A01 - A11) * (B1k0 + B1k1)
        U[0 * m + 1][col] = 1
        U[1 * m + 1][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1

    # Function to add a 2x1x4 schoolbook block (rank 8)
    # A indices: 0,1 (rows), 2 (col)
    # B indices: 2 (row), k0..k0+3 (cols)
    # C indices: 0,1 (rows), k0..k0+3 (cols)
    def add_2x1x4_block(k_start):
        nonlocal col
        for i in range(2): # A rows
            for k in range(4): # B cols
                U[i * m + 2][col] = 1
                V[2 * p + k_start + k][col] = 1
                W[(k_start + k) * n + i][col] = 1
                col += 1

    # Block 1: cols 0-3 of B
    add_2x2x4_block(0)
    add_2x1x4_block(0)
    
    # Block 2: cols 4-7 of B
    add_2x2x4_block(4)
    add_2x1x4_block(4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
