"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 22 decomposition
    # Split A (2x3) into A1 (2x2) and A2 (2x1)
    # Split B (3x4) into B1 (2x4) and B2 (1x4)
    # B1 split into B1a (2x2) and B1b (2x2)
    # Term 1: A1 * B1a (2x2x2) -> Strassen (7)
    # Term 2: A1 * B1b (2x2x2) -> Strassen (7)
    # Term 3: A2 * B2 (2x1x4) -> Schoolbook (8)
    
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen decomposition for 2x2x2
    # A indices: i in 0..1, j in 0..1 (offset by j_off)
    # B indices: j in 0..1 (offset by j_off), k in 0..1 (offset by k_off)
    # C indices: k in 0..1 (offset by k_off), i in 0..1
    def add_strassen(j_off, k_off):
        nonlocal col
        # U indices for A1: i*3 + (j_off + j)
        # V indices for B1a: (j_off + j)*4 + (k_off + k)
        # W indices for C: (k_off + k)*2 + i
        
        # m1 = (a00 + a11) * (b00 + b11)
        U[0 * 3 + j_off][col] = 1
        U[1 * 3 + (j_off + 1)][col] = 1
        V[(j_off + 0) * 4 + (k_off + 0)][col] = 1
        V[(j_off + 1) * 4 + (k_off + 1)][col] = 1
        W[(k_off + 0) * 2 + 0][col] = 1
        W[(k_off + 1) * 2 + 0][col] = 1
        W[(k_off + 0) * 2 + 1][col] = 1
        W[(k_off + 1) * 2 + 1][col] = 1
        col += 1
        
        # m2 = (a10 + a11) * b00
        U[1 * 3 + (j_off + 0)][col] = 1
        U[1 * 3 + (j_off + 1)][col] = 1
        V[(j_off + 0) * 4 + (k_off + 0)][col] = 1
        W[(k_off + 0) * 2 + 1][col] = 1
        W[(k_off + 1) * 2 + 1][col] = 1
        col += 1
        
        # m3 = (a00 - a10) * (b10 + b11)
        U[0 * 3 + j_off][col] = 1
        U[1 * 3 + (j_off + 0)][col] = -1
        V[(j_off + 1) * 4 + (k_off + 0)][col] = 1
        V[(j_off + 1) * 4 + (k_off + 1)][col] = 1
        W[(k_off + 0) * 2 + 0][col] = -1
        W[(k_off + 1) * 2 + 0][col] = -1
        W[(k_off + 0) * 2 + 1][col] = 1
        W[(k_off + 1) * 2 + 1][col] = 1
        col += 1
        
        # m4 = (a00 + a11) * (b11 - b01)
        U[0 * 3 + j_off][col] = 1
        U[1 * 3 + (j_off + 1)][col] = 1
        V[(j_off + 1) * 4 + (k_off + 1)][col] = 1
        V[(j_off + 0) * 4 + (k_off + 1)][col] = -1
        W[(k_off + 0) * 2 + 0][col] = -1
        W[(k_off + 1) * 2 + 0][col] = -1
        W[(k_off + 0) * 2 + 1][col] = 1
        W[(k_off + 1) * 2 + 1][col] = 1
        col += 1
        
        # m5 = (a00 + a10) * (b00 - b10)
        U[0 * 3 + j_off][col] = 1
        U[1 * 3 + (j_off + 0)][col] = 1
        V[(j_off + 0) * 4 + (k_off + 0)][col] = 1
        V[(j_off + 1) * 4 + (k_off + 0)][col] = -1
        W[(k_off + 0) * 2 + 0][col] = 1
        W[(k_off + 1) * 2 + 0][col] = 1
        W[(k_off + 0) * 2 + 1][col] = -1
        W[(k_off + 1) * 2 + 1][col] = -1
        col += 1
        
        # m6 = (a01 + a11) * (b01 - b00)
        U[0 * 3 + (j_off + 1)][col] = 1
        U[1 * 3 + (j_off + 1)][col] = 1
        V[(j_off + 0) * 4 + (k_off + 1)][col] = 1
        V[(j_off + 0) * 4 + (k_off + 0)][col] = -1
        W[(k_off + 0) * 2 + 0][col] = -1
        W[(k_off + 1) * 2 + 0][col] = -1
        W[(k_off + 0) * 2 + 1][col] = 1
        W[(k_off + 1) * 2 + 1][col] = 1
        col += 1
        
        # m7 = (a01 - a10) * (b10 + b11) -- Wait, standard Strassen m7 is (b-c)(g+h)
        # Let's use the standard mapping:
        # C00 = m1 - m4 + m5 - m7
        # C01 = m4 + m7
        # C10 = m2 + m3
        # C11 = m1 - m2 + m6 - m3
        # So m7 must be defined such that this works.
        # m7 = (a01 - a10) * (b10 + b11)
        U[0 * 3 + (j_off + 1)][col] = 1
        U[1 * 3 + (j_off + 0)][col] = -1
        V[(j_off + 1) * 4 + (k_off + 0)][col] = 1
        V[(j_off + 1) * 4 + (k_off + 1)][col] = 1
        W[(k_off + 0) * 2 + 0][col] = -1
        W[(k_off + 1) * 2 + 0][col] = -1
        W[(k_off + 0) * 2 + 1][col] = 1
        W[(k_off + 1) * 2 + 1][col] = 1
        col += 1

    # Add Strassen for A1 * B1a (j_off=0, k_off=0)
    add_strassen(0, 0)
    
    # Add Strassen for A1 * B1b (j_off=0, k_off=2)
    # Note: B1b is columns 2,3 of B. So k_off=2.
    add_strassen(0, 2)
    
    # Add Schoolbook for A2 * B2
    # A2 is row 0,1 col 2 (index 2, 5 in U)
    # B2 is row 2 (index 8,9,10,11 in V)
    # Output is all 8 elements
    for i in range(2):
        for k in range(4):
            U[i * 3 + 2][col] = 1
            V[2 * 4 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
