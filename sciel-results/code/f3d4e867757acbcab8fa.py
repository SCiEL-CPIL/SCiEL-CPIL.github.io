"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 36 decomposition
    # Split m (4) into two blocks of 2: J1={0,1}, J2={2,3}
    # Split p (5) into three blocks: K1={0,1}, K2={2}, K3={3,4}
    # For each J block, we compute:
    #   Strassen(2x2x2) for K1 (Rank 7)
    #   Schoolbook(2x2x1) for K2 (Rank 4)
    #   Strassen(2x2x2) for K3 (Rank 7)
    # Total per J block: 18. Total Rank: 36.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    def add_strassen_block(j_offset, k_offset):
        nonlocal col
        # A indices: i in {0,1}, j in {j_offset, j_offset+1}
        # B indices: j in {j_offset, j_offset+1}, k in {k_offset, k_offset+1}
        # C indices: i in {0,1}, k in {k_offset, k_offset+1}
        
        # M1 = (A00 + A11) * (B00 + B11)
        U[0 * m + j_offset][col] += 1
        U[1 * m + (j_offset + 1)][col] += 1
        V[j_offset * p + k_offset][col] += 1
        V[(j_offset + 1) * p + (k_offset + 1)][col] += 1
        W[k_offset * n + 0][col] += 1
        W[(k_offset + 1) * n + 1][col] += 1
        W[k_offset * n + 1][col] -= 1
        W[(k_offset + 1) * n + 0][col] -= 1
        col += 1
        
        # M2 = (A10 + A11) * B00
        U[1 * m + j_offset][col] += 1
        U[1 * m + (j_offset + 1)][col] += 1
        V[j_offset * p + k_offset][col] += 1
        W[k_offset * n + 1][col] += 1
        col += 1
        
        # M3 = A00 * (B10 - B11)
        U[0 * m + j_offset][col] += 1
        V[(j_offset + 1) * p + k_offset][col] += 1
        V[(j_offset + 1) * p + (k_offset + 1)][col] -= 1
        W[k_offset * n + 0][col] += 1
        W[(k_offset + 1) * n + 0][col] -= 1
        col += 1
        
        # M4 = A11 * (B01 - B00)
        U[1 * m + (j_offset + 1)][col] += 1
        V[j_offset * p + (k_offset + 1)][col] += 1
        V[j_offset * p + k_offset][col] -= 1
        W[(k_offset + 1) * n + 1][col] += 1
        W[k_offset * n + 1][col] -= 1
        col += 1
        
        # M5 = (A00 + A01) * B11
        U[0 * m + j_offset][col] += 1
        U[0 * m + (j_offset + 1)][col] += 1
        V[(j_offset + 1) * p + (k_offset + 1)][col] += 1
        W[(k_offset + 1) * n + 0][col] += 1
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[1 * m + j_offset][col] += 1
        U[0 * m + j_offset][col] -= 1
        V[j_offset * p + k_offset][col] += 1
        V[j_offset * p + (k_offset + 1)][col] += 1
        W[k_offset * n + 1][col] += 1
        W[(k_offset + 1) * n + 1][col] += 1
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[0 * m + (j_offset + 1)][col] += 1
        U[1 * m + (j_offset + 1)][col] -= 1
        V[(j_offset + 1) * p + k_offset][col] += 1
        V[(j_offset + 1) * p + (k_offset + 1)][col] += 1
        W[k_offset * n + 0][col] += 1
        W[(k_offset + 1) * n + 0][col] += 1
        col += 1

    # Helper to add a Schoolbook 2x2x1 block
    def add_schoolbook_block(j_offset, k_offset):
        nonlocal col
        # A indices: i in {0,1}, j in {j_offset, j_offset+1}
        # B indices: j in {j_offset, j_offset+1}, k in {k_offset}
        # C indices: i in {0,1}, k in {k_offset}
        
        # C00 += A00 * B00
        U[0 * m + j_offset][col] += 1
        V[j_offset * p + k_offset][col] += 1
        W[k_offset * n + 0][col] += 1
        col += 1
        
        # C01 += A01 * B00 (Wait, B is 2x1, so only one column k_offset)
        # Actually, for 2x2x1, B has 2 rows and 1 column.
        # C has 2 rows and 1 column.
        # C_i = A_i0 * B_0 + A_i1 * B_1
        
        # C0 += A00 * B_0 + A01 * B_1
        # C1 += A10 * B_0 + A11 * B_1
        
        # Term 1: A00 * B_0
        U[0 * m + j_offset][col] += 1
        V[j_offset * p + k_offset][col] += 1
        W[k_offset * n + 0][col] += 1
        col += 1
        
        # Term 2: A01 * B_1
        U[0 * m + (j_offset + 1)][col] += 1
        V[(j_offset + 1) * p + k_offset][col] += 1
        W[k_offset * n + 0][col] += 1
        col += 1
        
        # Term 3: A10 * B_0
        U[1 * m + j_offset][col] += 1
        V[j_offset * p + k_offset][col] += 1
        W[k_offset * n + 1][col] += 1
        col += 1
        
        # Term 4: A11 * B_1
        U[1 * m + (j_offset + 1)][col] += 1
        V[(j_offset + 1) * p + k_offset][col] += 1
        W[k_offset * n + 1][col] += 1
        col += 1

    # Process Block J1 (j=0,1)
    add_strassen_block(0, 0)   # k=0,1
    add_schoolbook_block(0, 2) # k=2
    add_strassen_block(0, 3)   # k=3,4
    
    # Process Block J2 (j=2,3)
    add_strassen_block(2, 0)   # k=0,1
    add_schoolbook_block(2, 2) # k=2
    add_strassen_block(2, 3)   # k=3,4
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
