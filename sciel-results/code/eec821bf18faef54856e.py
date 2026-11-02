"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (14, rank)
    # W: (p*n, rank) = (14, rank)
    
    # We will use rank 25
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(k_offset, r_offset):
        # k_offset: 0, 2, 4
        # r_offset: 0, 7, 14
        # Local k indices 0, 1 map to global k_offset + 0, k_offset + 1
        # U indices: i*m + j. i in 0,1, j in 0,1.
        # V indices: j*p + k. j in 0,1, k in k_offset, k_offset+1.
        # W indices: k*n + i. k in k_offset, k_offset+1, i in 0,1.
        
        # M1: (a00+a11)(b00+b11) -> c00, c11
        r = r_offset
        U[0][r] += 1; U[3][r] += 1
        V[k_offset][r] += 1; V[7 + k_offset][r] += 1
        W[k_offset * n + 0][r] += 1
        W[(k_offset + 1) * n + 1][r] += 1
        
        # M2: (a10+a11)b00 -> c10, c11(-)
        r = r_offset + 1
        U[2][r] += 1; U[3][r] += 1
        V[k_offset][r] += 1
        W[k_offset * n + 1][r] += 1
        W[(k_offset + 1) * n + 1][r] += -1
        
        # M3: a00(b01-b11) -> c01, c11
        r = r_offset + 2
        U[0][r] += 1
        V[k_offset + 1][r] += 1; V[7 + k_offset + 1][r] += -1
        W[(k_offset + 1) * n + 0][r] += 1
        W[(k_offset + 1) * n + 1][r] += 1
        
        # M4: a11(b10-b00) -> c00, c10
        r = r_offset + 3
        U[3][r] += 1
        V[7 + k_offset][r] += 1; V[k_offset][r] += -1
        W[k_offset * n + 0][r] += 1
        W[k_offset * n + 1][r] += 1
        
        # M5: (a00+a01)b11 -> c00(-), c01
        r = r_offset + 4
        U[0][r] += 1; U[1][r] += 1
        V[7 + k_offset + 1][r] += 1
        W[k_offset * n + 0][r] += -1
        W[(k_offset + 1) * n + 0][r] += 1
        
        # M6: (a10-a00)(b00+b01) -> c11
        r = r_offset + 5
        U[2][r] += 1; U[0][r] += -1
        V[k_offset][r] += 1; V[k_offset + 1][r] += 1
        W[(k_offset + 1) * n + 1][r] += 1
        
        # M7: (a01-a11)(b10+b11) -> c00
        r = r_offset + 6
        U[1][r] += 1; U[3][r] += -1
        V[7 + k_offset][r] += 1; V[7 + k_offset + 1][r] += 1
        W[k_offset * n + 0][r] += 1

    # Add 3 Strassen blocks for k=0,1; 2,3; 4,5
    add_strassen_block(0, 0)
    add_strassen_block(2, 7)
    add_strassen_block(4, 14)
    
    # Add Schoolbook block for k=6
    r_offset = 21
    # C_06 = A_00 B_06 + A_01 B_16
    r = r_offset
    U[0][r] += 1
    V[6][r] += 1
    W[6 * n + 0][r] += 1
    
    r = r_offset + 1
    U[1][r] += 1
    V[13][r] += 1
    W[6 * n + 0][r] += 1
    
    # C_16 = A_10 B_06 + A_11 B_16
    r = r_offset + 2
    U[2][r] += 1
    V[6][r] += 1
    W[6 * n + 1][r] += 1
    
    r = r_offset + 3
    U[3][r] += 1
    V[13][r] += 1
    W[6 * n + 1][r] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
