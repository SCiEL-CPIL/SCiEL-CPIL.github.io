"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will use Strassen for 7 pairs of columns (14 columns) and naive for the last 1 column.
    # Strassen rank for 2x2x2 is 7.
    # Naive rank for 2x2x1 is 4.
    # Total rank = 7 * 7 + 4 = 53.
    
    rank = 7 * 7 + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add a Strassen block for columns k, k+1
    def add_strassen(k1, k2, start_rank):
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A12+A22)B11
        # M3 = A11(B22-B11)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Indices for A: i*m+j
        # Indices for B: j*p+k
        # Indices for C: k*n+i
        
        terms = [
            # (U_indices, V_indices, W_indices)
            # M1
            ([(0,1), (1,1)], [(0,0), (1,1)], [(0,0), (1,1)]), # (A11+A22)(B11+B22) -> C11, C22
            # M2
            ([(0,1), (1,1)], [(0,0)], [(0,1)]), # (A12+A22)B11 -> C21? No.
            # Let's map carefully.
        ]
        
        # Correct Strassen mapping:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # M1 = (A11+A22)(B11+B22)
        # U: A11, A22. V: B11, B22. W: C11, C22.
        U[0*m+0][start_rank+0] += 1 # A11
        U[1*m+1][start_rank+0] += 1 # A22
        V[0*p+k1][start_rank+0] += 1 # B11
        V[1*p+k2][start_rank+0] += 1 # B22
        W[k1*n+0][start_rank+0] += 1 # C11
        W[k2*n+1][start_rank+0] += 1 # C22
        
        # M2 = (A12+A22)B11
        # U: A12, A22. V: B11. W: C21.
        U[0*m+1][start_rank+1] += 1 # A12
        U[1*m+1][start_rank+1] += 1 # A22
        V[0*p+k1][start_rank+1] += 1 # B11
        W[k1*n+1][start_rank+1] += 1 # C21
        
        # M3 = A11(B22-B11)
        # U: A11. V: B22, -B11. W: C12.
        U[0*m+0][start_rank+2] += 1 # A11
        V[1*p+k2][start_rank+2] += 1 # B22
        V[0*p+k1][start_rank+2] -= 1 # -B11
        W[k2*n+0][start_rank+2] += 1 # C12
        
        # M4 = A22(B12-B11)
        # U: A22. V: B12, -B11. W: C11, C21.
        U[1*m+1][start_rank+3] += 1 # A22
        V[0*p+k2][start_rank+3] += 1 # B12
        V[0*p+k1][start_rank+3] -= 1 # -B11
        W[k1*n+0][start_rank+3] += 1 # C11
        W[k1*n+1][start_rank+3] += 1 # C21
        
        # M5 = (A11+A12)B22
        # U: A11, A12. V: B22. W: -C11, C12.
        U[0*m+0][start_rank+4] += 1 # A11
        U[0*m+1][start_rank+4] += 1 # A12
        V[1*p+k2][start_rank+4] += 1 # B22
        W[k1*n+0][start_rank+4] -= 1 # -C11
        W[k2*n+0][start_rank+4] += 1 # C12
        
        # M6 = (A21-A11)(B11+B12)
        # U: A21, -A11. V: B11, B12. W: C22.
        U[1*m+0][start_rank+5] += 1 # A21
        U[0*m+0][start_rank+5] -= 1 # -A11
        V[0*p+k1][start_rank+5] += 1 # B11
        V[0*p+k2][start_rank+5] += 1 # B12
        W[k2*n+1][start_rank+5] += 1 # C22
        
        # M7 = (A12-A22)(B21+B22)
        # U: A12, -A22. V: B21, B22. W: C11.
        U[0*m+1][start_rank+6] += 1 # A12
        U[1*m+1][start_rank+6] -= 1 # -A22
        V[1*p+k1][start_rank+6] += 1 # B21
        V[1*p+k2][start_rank+6] += 1 # B22
        W[k1*n+0][start_rank+6] += 1 # C11

    # Add 7 Strassen blocks
    for i in range(7):
        k1 = 2 * i
        k2 = 2 * i + 1
        add_strassen(k1, k2, i * 7)
        
    # Add 1 Naive block for the last column (k=14)
    # C1 = A11 B1 + A12 B2
    # C2 = A21 B1 + A22 B2
    k = 14
    start = 49
    
    # Term 1: A11 B1 -> C1
    U[0*m+0][start+0] = 1
    V[0*p+k][start+0] = 1
    W[k*n+0][start+0] = 1
    
    # Term 2: A12 B2 -> C1
    U[0*m+1][start+1] = 1
    V[1*p+k][start+1] = 1
    W[k*n+0][start+1] = 1
    
    # Term 3: A21 B1 -> C2
    U[1*m+0][start+2] = 1
    V[0*p+k][start+2] = 1
    W[k*n+1][start+2] = 1
    
    # Term 4: A22 B2 -> C2
    U[1*m+1][start+3] = 1
    V[1*p+k][start+3] = 1
    W[k*n+1][start+3] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
