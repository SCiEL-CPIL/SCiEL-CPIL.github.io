"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+c)h
    # P6 = (d-b)(e+g)
    # P7 = (b-d)(f+h)
    
    # Mapping to U, V, W for block (i,j,k)
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    def add_strassen_block(i_offset, j_offset, k_offset, base_col):
        # Local indices 0,1 for i,j,k
        # Global indices
        # i in {i_offset, i_offset+1}
        # j in {j_offset, j_offset+1}
        # k in {k_offset, k_offset+1}
        
        # P1: (A00+A11)(B00+B11)
        # U: (0,0)=1, (1,1)=1
        # V: (0,0)=1, (1,1)=1
        # W: (0,0)=1, (1,1)=1
        U[(i_offset)*m + (j_offset)][base_col] = 1
        U[(i_offset+1)*m + (j_offset+1)][base_col] = 1
        V[(j_offset)*p + (k_offset)][base_col] = 1
        V[(j_offset+1)*p + (k_offset+1)][base_col] = 1
        W[(k_offset)*n + (i_offset)][base_col] = 1
        W[(k_offset+1)*n + (i_offset+1)][base_col] = 1
        col = base_col + 1
        
        # P2: (A10+A11)B00
        # U: (1,0)=1, (1,1)=1
        # V: (0,0)=1
        # W: (0,1)=1
        U[(i_offset+1)*m + (j_offset)][col] = 1
        U[(i_offset+1)*m + (j_offset+1)][col] = 1
        V[(j_offset)*p + (k_offset)][col] = 1
        W[(k_offset)*n + (i_offset+1)][col] = 1
        col += 1
        
        # P3: A00(B10-B11)
        # U: (0,0)=1
        # V: (1,0)=1, (1,1)=-1
        # W: (0,0)=1
        U[(i_offset)*m + (j_offset)][col] = 1
        V[(j_offset+1)*p + (k_offset)][col] = 1
        V[(j_offset+1)*p + (k_offset+1)][col] = -1
        W[(k_offset)*n + (i_offset)][col] = 1
        col += 1
        
        # P4: A11(B01-B00)
        # U: (1,1)=1
        # V: (0,1)=1, (0,0)=-1
        # W: (1,1)=1
        U[(i_offset+1)*m + (j_offset+1)][col] = 1
        V[(j_offset)*p + (k_offset+1)][col] = 1
        V[(j_offset)*p + (k_offset)][col] = -1
        W[(k_offset+1)*n + (i_offset+1)][col] = 1
        col += 1
        
        # P5: (A00+A10)B11
        # U: (0,0)=1, (1,0)=1
        # V: (1,1)=1
        # W: (1,0)=1
        U[(i_offset)*m + (j_offset)][col] = 1
        U[(i_offset+1)*m + (j_offset)][col] = 1
        V[(j_offset+1)*p + (k_offset+1)][col] = 1
        W[(k_offset+1)*n + (i_offset)][col] = 1
        col += 1
        
        # P6: (A11-A01)(B00+B10)
        # U: (1,1)=1, (0,1)=-1
        # V: (0,0)=1, (1,0)=1
        # W: (0,1)=1
        U[(i_offset+1)*m + (j_offset+1)][col] = 1
        U[(i_offset)*m + (j_offset+1)][col] = -1
        V[(j_offset)*p + (k_offset)][col] = 1
        V[(j_offset+1)*p + (k_offset)][col] = 1
        W[(k_offset)*n + (i_offset+1)][col] = 1
        col += 1
        
        # P7: (A01-A11)(B11+B01)
        # U: (0,1)=1, (1,1)=-1
        # V: (1,1)=1, (0,1)=1
        # W: (1,0)=1
        U[(i_offset)*m + (j_offset+1)][col] = 1
        U[(i_offset+1)*m + (j_offset+1)][col] = -1
        V[(j_offset+1)*p + (k_offset+1)][col] = 1
        V[(j_offset)*p + (k_offset+1)][col] = 1
        W[(k_offset+1)*n + (i_offset)][col] = 1
        col += 1
        
        return col

    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    col = add_strassen_block(0, 0, 0, col)
    
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3}
    col = add_strassen_block(0, 0, 2, col)
    
    # Block 3: i in {0,1}, j in {0,1}, k in {4}
    # Schoolbook 2x2x1
    for i in range(2):
        for j in range(2):
            k = 4
            U[i*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
            
    # Block 4: i in {0,1}, j in {2}, k in {0..4}
    # Schoolbook 2x1x5
    for i in range(2):
        for k in range(5):
            j = 2
            U[i*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
