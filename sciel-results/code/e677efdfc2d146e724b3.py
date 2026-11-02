"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a decomposition of rank 11:
    # 7 products for Strassen on the first 2 columns of B (2x2 * 2x2)
    # 4 products for schoolbook on the 3rd column of B (2x2 * 2x1)
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a product term
    # U_idx: index in U (i*m+j)
    # V_idx: index in V (j*p+k)
    # W_idx: index in W (k*n+i)
    # col: column index in the decomposition
    def add_term(u_idx, v_idx, w_idx, col, val=1):
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val

    # Strassen for 2x2 * 2x2 (columns 0,1 of B)
    # P1 = (A00+A11)(B00+B11) -> C00, C11
    # P2 = (A10+A11)B00 -> C10
    # P3 = A00(B01-B11) -> C01
    # P4 = A11(B10-B00) -> C10
    # P5 = (A00+A01)B11 -> C01
    # P6 = (A10-A00)(B00+B01) -> C10, C11
    # P7 = (A01-A11)(B10+B11) -> C01, C11
    
    # Note: Indices are 0-based.
    # A indices: 00, 01, 10, 11 -> U idx 0, 1, 2, 3
    # B indices (for p=2 part): 00, 01, 10, 11 -> V idx 0, 1, 2, 3 (since j*p+k, p=2)
    # C indices: 00, 01, 10, 11 -> W idx 0, 1, 2, 3 (since k*n+i, n=2)
    
    # P1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1 # C10
    
    # P3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1 # B01 - B11
    W[2][2] = 1 # C01
    
    # P4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1 # B10 - B00
    W[1][3] = 1 # C10
    
    # P5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[2][4] = 1 # C01
    
    # P6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[1][5] = 1; W[3][5] = 1 # C10, C11
    
    # P7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[2][6] = 1; W[3][6] = 1 # C01, C11
    
    # Schoolbook for 2x2 * 2x1 (column 2 of B)
    # B indices: 02, 12 -> V idx 0*3+2=2, 1*3+2=5
    # C indices: 02, 12 -> W idx 2*2+0=4, 2*2+1=5
    
    # P8 = A00 * B02 -> C02
    U[0][7] = 1
    V[2][7] = 1
    W[4][7] = 1
    
    # P9 = A01 * B12 -> C02
    U[1][8] = 1
    V[5][8] = 1
    W[4][8] = 1
    
    # P10 = A10 * B02 -> C12
    U[2][9] = 1
    V[2][9] = 1
    W[5][9] = 1
    
    # P11 = A11 * B12 -> C12
    U[3][10] = 1
    V[5][10] = 1
    W[5][10] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
