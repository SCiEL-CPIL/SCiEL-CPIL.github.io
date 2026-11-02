"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-32 tensor decomposition for 2x2 * 2x9 matrix multiplication.
    
    Strategy:
    - The rank of the 2x2x2 matrix multiplication tensor is 7 (Strassen).
    - The rank of the 2x2x1 matrix multiplication tensor is 4 (Schoolbook).
    - We partition the 9 output columns into four pairs and one single column.
    - For each pair of columns, we apply the Strassen decomposition (rank 7).
    - For the single column, we apply the Schoolbook decomposition (rank 4).
    - Total rank: 4 * 7 + 4 = 32.
    
    This is the minimal rank for this problem, as R(2,2,p) = ceil(7p/2) for small p,
    and ceil(7*9/2) = 32.
    
    Index conventions:
    - U has shape (n*m, rank) = (4, rank), indexed by (i*m + j) where i in {0,1}, j in {0,1}
    - V has shape (m*p, rank) = (18, rank), indexed by (j*p + k) where j in {0,1}, k in {0..8}
    - W has shape (p*n, rank) = (18, rank), indexed by (k*n + i) where k in {0..8}, i in {0,1}
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)]   # 4 x 32
    V = [[0] * rank for _ in range(m * p)]   # 18 x 32
    W = [[0] * rank for _ in range(p * n)]   # 18 x 32
    
    def strassen_block(k1, k2, start_col):
        """Add 7 Strassen products for columns k1, k2 starting at column start_col."""
        nonlocal U, V, W
        # V row indices: j*p + k
        v_00 = k1       # b_{0,k1}
        v_01 = k2       # b_{0,k2}
        v_10 = p + k1   # b_{1,k1}
        v_11 = p + k2   # b_{1,k2}
        
        # W row indices: k*n + i
        w_c00 = k1 * n + 0  # C[0,k1]
        w_c10 = k1 * n + 1  # C[1,k1]
        w_c01 = k2 * n + 0  # C[0,k2]
        w_c11 = k2 * n + 1  # C[1,k2]
        
        # U row indices: i*m + j
        u_00 = 0  # a_{0,0}
        u_01 = 1  # a_{0,1}
        u_10 = 2  # a_{1,0}
        u_11 = 3  # a_{1,1}
        
        # M1 = (a00 + a11) * (b0,k1 + b1,k2)
        c = start_col
        U[u_00][c] += 1; U[u_11][c] += 1
        V[v_00][c] += 1; V[v_11][c] += 1
        W[w_c00][c] += 1; W[w_c11][c] += 1
        
        # M2 = (a10 + a11) * b0,k1
        c = start_col + 1
        U[u_10][c] += 1; U[u_11][c] += 1
        V[v_00][c] += 1
        W[w_c10][c] += 1; W[w_c11][c] -= 1
        
        # M3 = a00 * (b0,k2 - b1,k2)
        c = start_col + 2
        U[u_00][c] += 1
        V[v_01][c] += 1; V[v_11][c] -= 1
        W[w_c01][c] += 1; W[w_c11][c] += 1
        
        # M4 = a11 * (b1,k1 - b0,k1)
        c = start_col + 3
        U[u_11][c] += 1
        V[v_10][c] += 1; V[v_00][c] -= 1
        W[w_c00][c] += 1; W[w_c10][c] += 1
        
        # M5 = (a00 + a01) * b1,k2
        c = start_col + 4
        U[u_00][c] += 1; U[u_01][c] += 1
        V[v_11][c] += 1
        W[w_c00][c] -= 1; W[w_c01][c] += 1
        
        # M6 = (a10 - a00) * (b0,k1 + b0,k2)
        c = start_col + 5
        U[u_10][c] += 1; U[u_00][c] -= 1
        V[v_00][c] += 1; V[v_01][c] += 1
        W[w_c11][c] += 1
        
        # M7 = (a01 - a11) * (b1,k1 + b1,k2)
        c = start_col + 6
        U[u_01][c] += 1; U[u_11][c] -= 1
        V[v_10][c] += 1; V[v_11][c] += 1
        W[w_c00][c] += 1
        
        return start_col + 7
    
    def schoolbook_single(k, start_col):
        """Add 4 schoolbook products for a single column k."""
        nonlocal U, V, W
        v_0 = k       # b_{0,k}
        v_1 = p + k   # b_{1,k}
        w_0 = k * n   # C[0,k]
        w_1 = k * n + 1  # C[1,k]
        
        c = start_col
        U[0][c] += 1; V[v_0][c] += 1; W[w_0][c] += 1  # a00*b0k
        
        c = start_col + 1
        U[1][c] += 1; V[v_1][c] += 1; W[w_0][c] += 1  # a01*b1k
        
        c = start_col + 2
        U[2][c] += 1; V[v_0][c] += 1; W[w_1][c] += 1  # a10*b0k
        
        c = start_col + 3
        U[3][c] += 1; V[v_1][c] += 1; W[w_1][c] += 1  # a11*b1k
        
        return start_col + 4
    
    col = 0
    # Apply Strassen to pairs (0,1), (2,3), (4,5), (6,7)
    for k1, k2 in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        col = strassen_block(k1, k2, col)
        
    # Apply Schoolbook to single column 8
    col = schoolbook_single(8, col)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
