"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A11)(B11+B00) -> This is M1. 
    # Let's use the standard 7.
    
    # I will use the explicit tensor entries for Strassen.
    # T = sum r U_r V_r W_r^T
    
    # For each of the 4 column blocks of B (size 2)
    for r in range(4):
        p_off = r * 2
        
        # Part 1: A_2x2 * B_2x2 (Strassen, rank 7)
        # Local indices i,j,k in 0..1
        # Global i: i, Global j: j, Global k: p_off + k
        
        # M1
        U[0][col] += 1; U[3][col] += 1 # A00, A11 (j=0,1) -> idx 0, 3? 
        # U index: i*m + j. m=3.
        # A is 2x3. A_2x2 is cols 0,1.
        # A00: i=0, j=0 -> 0*3+0=0.
        # A11: i=1, j=1 -> 1*3+1=4.
        # A10: i=1, j=0 -> 1*3+0=3.
        # A01: i=0, j=1 -> 0*3+1=1.
        
        # Correct U indices for A_2x2:
        # A00: 0
        # A01: 1
        # A10: 3
        # A11: 4
        
        # V indices for B_2x2 (rows 0,1 of block, cols p_off, p_off+1)
        # B is 3x8. B_2x2 is rows 0,1.
        # B00: j=0, k=p_off -> 0*8 + p_off
        # B01: j=0, k=p_off+1 -> 0*8 + p_off+1
        # B10: j=1, k=p_off -> 1*8 + p_off
        # B11: j=1, k=p_off+1 -> 1*8 + p_off+1
        
        # W indices for C_2x2 (cols p_off, p_off+1)
        # C is 8x2.
        # C00: k=p_off, i=0 -> p_off*2 + 0
        # C01: k=p_off+1, i=0 -> (p_off+1)*2 + 0
        # C10: k=p_off, i=1 -> p_off*2 + 1
        # C11: k=p_off+1, i=1 -> (p_off+1)*2 + 1
        
        # Strassen M1
        U[0][col] += 1
        U[4][col] += 1
        V[p_off][col] += 1
        V[8 + p_off][col] += 1
        W[p_off * 2][col] += 1
        W[(p_off + 1) * 2][col] += 1
        W[p_off * 2 + 1][col] += 1
        W[(p_off + 1) * 2 + 1][col] += 1
        col += 1
        
        # Strassen M2
        U[3][col] += 1
        U[4][col] += 1
        V[p_off][col] += 1
        W[p_off * 2 + 1][col] += 1
        col += 1
        
        # Strassen M3
        U[0][col] += 1
        V[p_off + 1][col] += 1
        V[8 + p_off + 1][col] -= 1
        W[(p_off + 1) * 2][col] += 1
        W[(p_off + 1) * 2 + 1][col] -= 1
        col += 1
        
        # Strassen M4
        U[4][col] += 1
        V[8 + p_off][col] += 1
        V[p_off][col] -= 1
        W[p_off * 2][col] -= 1
        W[p_off * 2 + 1][col] += 1
        col += 1
        
        # Strassen M5
        U[0][col] += 1
        U[4][col] += 1
        V[8 + p_off][col] += 1
        V[p_off][col] += 1
        W[p_off * 2][col] += 1
        W[(p_off + 1) * 2 + 1][col] += 1
        col += 1
        
        # Strassen M6
        U[1][col] += 1
        U[4][col] -= 1
        V[8 + p_off][col] += 1
        V[8 + p_off + 1][col] += 1
        W[(p_off + 1) * 2 + 1][col] -= 1
        col += 1
        
        # Strassen M7
        U[0][col] += 1
        U[3][col] -= 1
        V[p_off][col] += 1
        V[p_off + 1][col] += 1
        W[p_off * 2][col] -= 1
        W[(p_off + 1) * 2][col] += 1
        col += 1
        
        # Part 2: A_2x1 * B_1x2 (Schoolbook, rank 4)
        # A col 2 (index 2 in j).
        # A02: i=0, j=2 -> 0*3+2=2
        # A12: i=1, j=2 -> 1*3+2=5
        
        # B row 2 (index 2 in j).
        # B20: j=2, k=p_off -> 2*8 + p_off
        # B21: j=2, k=p_off+1 -> 2*8 + p_off+1
        
        # C same as above.
        
        # Term 1: A02 * B20 -> C00
        U[2][col] += 1
        V[16 + p_off][col] += 1
        W[p_off * 2][col] += 1
        col += 1
        
        # Term 2: A02 * B21 -> C01
        U[2][col] += 1
        V[16 + p_off + 1][col] += 1
        W[(p_off + 1) * 2][col] += 1
        col += 1
        
        # Term 3: A12 * B20 -> C10
        U[5][col] += 1
        V[16 + p_off][col] += 1
        W[p_off * 2 + 1][col] += 1
        col += 1
        
        # Term 4: A12 * B21 -> C11
        U[5][col] += 1
        V[16 + p_off + 1][col] += 1
        W[(p_off + 1) * 2 + 1][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
