"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Split: A = [A2x2 | A2x1], B = [B2x6; B1x6]
    # C = A2x2 * B2x6 + A2x1 * B1x6
    # A2x2*B2x6: split B2x6 into three 2x2 blocks, each uses Strassen (rank 7)
    # A2x1*B1x6: 6 scalar mults (rank 6)
    # Total rank: 3*7 + 6 = 27
    
    rank = 27
    U = [[0] * rank for _ in range(n * m)]  # 2*3 = 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 3*6 = 18 rows
    W = [[0] * rank for _ in range(p * n)]  # 6*2 = 12 rows
    
    col = 0
    # Strassen decomposition for 2x2 * 2x2
    # C = A*B where A, B are 2x2
    # A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
    # Strassen:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b10)
    # M7 = (a01-a11)*(b01+b11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    def strassen_2x2(base_col):
        """Add 7 Strassen factors for one 2x2*2x2 block.
        A indices: rows i in {0,1}, cols j in {0,1} -> U[i*3 + j] (since m=3, j=0,1 for first 2 cols)
        B indices: rows j in {0,1}, cols k offset by k_off -> V[j*6 + k_off + dk]
        C indices: rows i in {0,1}, cols k offset by k_off -> W[(k_off+dk)*2 + i]
        """
        # M1 = (a00+a11)*(b00+b11)
        U[0*3+0][base_col+0] = 1  # a00
        U[1*3+1][base_col+0] = 1  # a11
        V[0*6+0][base_col+0] = 1  # b00 (relative)
        V[1*6+1][base_col+0] = 1  # b11 (relative)
        W[0*2+0][base_col+0] = 1  # c00
        W[1*2+1][base_col+0] = 1  # c11
        
        # M2 = (a10+a11)*b00
        U[1*3+0][base_col+1] = 1  # a10
        U[1*3+1][base_col+1] = 1  # a11
        V[0*6+0][base_col+1] = 1  # b00
        W[1*2+0][base_col+1] = 1  # c10
        
        # M3 = a00*(b01-b11)
        U[0*3+0][base_col+2] = 1  # a00
        V[0*6+1][base_col+2] = 1  # b01
        V[1*6+1][base_col+2] = -1 # b11
        W[0*2+1][base_col+2] = 1  # c01
        
        # M4 = a11*(b10-b00)
        U[1*3+1][base_col+3] = 1  # a11
        V[1*6+0][base_col+3] = 1  # b10
        V[0*6+0][base_col+3] = -1 # b00
        W[0*2+0][base_col+3] = 1  # c00
        W[1*2+0][base_col+3] = 1  # c10
        
        # M5 = (a00+a01)*b11
        U[0*3+0][base_col+4] = 1  # a00
        U[0*3+1][base_col+4] = 1  # a01
        V[1*6+1][base_col+4] = 1  # b11
        W[0*2+0][base_col+4] = -1 # c00
        W[0*2+1][base_col+4] = 1  # c01
        
        # M6 = (a10-a00)*(b00+b10)
        U[1*3+0][base_col+5] = 1  # a10
        U[0*3+0][base_col+5] = -1 # a00
        V[0*6+0][base_col+5] = 1  # b00
        V[1*6+0][base_col+5] = 1  # b10
        W[1*2+1][base_col+5] = 1  # c11
        
        # M7 = (a01-a11)*(b01+b11)
        U[0*3+1][base_col+6] = 1  # a01
        U[1*3+1][base_col+6] = -1 # a11
        V[0*6+1][base_col+6] = 1  # b01
        V[1*6+1][base_col+6] = 1  # b11
        W[0*2+0][base_col+6] = 1  # c00
        W[1*2+1][base_col+6] = 1  # c11
        
        return base_col + 7
    
    # Three 2x2*2x2 blocks for A2x2 * B2x6
    # Block 0: A[:,0:2] * B[0:2, 0:2]
    # Block 1: A[:,0:2] * B[0:2, 2:4]
    # Block 2: A[:,0:2] * B[0:2, 4:6]
    for block in range(3):
        k_off = block * 2
        # Override V and W indices for this block's k offset
        # We need to adjust the V and W indices to use k_off
        
        # Let me redo this more carefully with explicit k offsets
        pass
    
    # Let me redo with proper k offsets
    col = 0
    for block in range(3):
        k_off = block * 2
        # M1 = (a00+a11)*(b00+b11)
        U[0][col] = 1; U[4][col] = 1  # a00, a11 (rows 0,1 of A, cols 0,1 -> indices 0,1,3,4 in U since m=3)
        # Wait, U has n*m = 6 rows. A[i,j] maps to U[i*m+j] = U[i*3+j]
        # A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[3], A[1,1]=U[4]
        # B[j,k] maps to V[j*p+k] = V[j*6+k]
        # B[0,k_off]=V[k_off], B[0,k_off+1]=V[k_off+1]
        # B[1,k_off]=V[6+k_off], B[1,k_off+1]=V[6+k_off+1]
        # C[i,k] maps to W[k*n+i] = W[k*2+i]
        # C[0,k_off]=W[2*k_off], C[0,k_off+1]=W[2*k_off+1]
        # C[1,k_off]=W[2*k_off+2], C[1,k_off+1]=W[2*k_off+3]
        
        # M1
        U[0][col] = 1; U[4][col] = 1
        V[k_off][col] = 1; V[6+k_off+1][col] = 1
        W[2*k_off][col] = 1; W[2*k_off+3][col] = 1
        col += 1
        
        # M2 = (a10+a11)*b00
        U[3][col] = 1; U[4][col] = 1
        V[k_off][col] = 1
        W[2*k_off+2][col] = 1
        col += 1
        
        # M3 = a00*(b01-b11)
        U[0][col] = 1
        V[k_off+1][col] = 1; V[6+k_off+1][col] = -1
        W[2*k_off+1][col] = 1
        col += 1
        
        # M4 = a11*(b10-b00)
        U[4][col] = 1
        V[6+k_off][col] = 1; V[k_off][col] = -1
        W[2*k_off][col] = 1; W[2*k_off+2][col] = 1
        col += 1
        
        # M5 = (a00+a01)*b11
        U[0][col] = 1; U[1][col] = 1
        V[6+k_off+1][col] = 1
        W[2*k_off][col] = -1; W[2*k_off+1][col] = 1
        col += 1
        
        # M6 = (a10-a00)*(b00+b10)
        U[3][col] = 1; U[0][col] = -1
        V[k_off][col] = 1; V[6+k_off][col] = 1
        W[2*k_off+3][col] = 1
        col += 1
        
        # M7 = (a01-a11)*(b01+b11)
        U[1][col] = 1; U[4][col] = -1
        V[k_off+1][col] = 1; V[6+k_off+1][col] = 1
        W[2*k_off][col] = 1; W[2*k_off+3][col] = 1
        col += 1
    
    # Now A2x1 * B1x6: A[0,2]*B[2,k] and A[1,2]*B[2,k]
    # A[0,2] = U[2], A[1,2] = U[5]
    # B[2,k] = V[12+k]
    # C[0,k] = W[2k], C[1,k] = W[2k+1]
    for k in range(6):
        # C[0,k] += A[0,2]*B[2,k]
        U[2][col] = 1
        V[12+k][col] = 1
        W[2*k][col] = 1
        col += 1
        # C[1,k] += A[1,2]*B[2,k]
        U[5][col] = 1
        V[12+k][col] = 1
        W[2*k+1][col] = 1
        col += 1
    
    # col should be 21 + 6 = 27
    assert col == rank, f"col={col}, rank={rank}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
