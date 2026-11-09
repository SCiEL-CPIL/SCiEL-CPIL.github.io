"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2x2():
    """
    Return the Strassen decomposition for 2x2x2 matrix multiplication.
    Rank 7 instead of 8.
    
    Returns U, V, W as lists of lists where:
    - U has shape (4, 7): U[i*2+j, r] for i,j in 0..1
    - V has shape (4, 7): V[j*2+k, r] for j,k in 0..1  
    - W has shape (4, 7): W[k*2+i, r] for k,i in 0..1
    
    The decomposition satisfies:
    C[i,k] = sum_j sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
    """
    # Strassen's 7 products:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    # For each product Mr, we define:
    # U[:, r] = coefficients for A entries
    # V[:, r] = coefficients for B entries  
    # W[:, r] = coefficients for C entries
    
    # U[i*2+j, r] = coefficient of a_{i,j} in the first factor of M_r
    # V[j*2+k, r] = coefficient of b_{j,k} in the second factor of M_r
    # W[k*2+i, r] = coefficient of c_{i,k} from M_r
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a11+a22)(b11+b22)
    # A part: a11=1, a22=1 -> U[0,0]=1, U[3,0]=1
    # B part: b11=1, b22=1 -> V[0,0]=1, V[3,0]=1
    # C part: c11+=1, c22+=1 -> W[0,0]=1, W[3,0]=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (a21+a22)b11
    # A part: a21=1, a22=1 -> U[2,1]=1, U[3,1]=1
    # B part: b11=1 -> V[0,1]=1
    # C part: c21+=1 -> W[1,1]=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    # M3 = a11(b12-b22)
    # A part: a11=1 -> U[0,2]=1
    # B part: b12=1, b22=-1 -> V[1,2]=1, V[3,2]=-1
    # C part: c12+=1 -> W[2,2]=1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1
    
    # M4 = a22(b21-b11)
    # A part: a22=1 -> U[3,3]=1
    # B part: b21=1, b11=-1 -> V[2,3]=1, V[0,3]=-1
    # C part: c11+=1, c21+=1 -> W[0,3]=1, W[1,3]=1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (a11+a12)b22
    # A part: a11=1, a12=1 -> U[0,4]=1, U[1,4]=1
    # B part: b22=1 -> V[3,4]=1
    # C part: c11-=1, c12+=1 -> W[0,4]=-1, W[2,4]=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (a21-a11)(b11+b12)
    # A part: a21=1, a11=-1 -> U[2,5]=1, U[0,5]=-1
    # B part: b11=1, b12=1 -> V[0,5]=1, V[1,5]=1
    # C part: c22+=1 -> W[3,5]=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (a12-a22)(b21+b22)
    # A part: a12=1, a22=-1 -> U[1,6]=1, U[3,6]=-1
    # B part: b21=1, b22=1 -> V[2,6]=1, V[3,6]=1
    # C part: c11+=1 -> W[0,6]=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _schoolbook_block(n, m, p):
    """
    Return schoolbook decomposition for nxmxp matrix multiplication.
    Rank = n*m*p
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n=2, m=5, p=6, max_check=10):
    """Verify decomposition using random integer matrices."""
    if not (len(U) == n*m and len(V) == m*p and len(W) == p*n):
        return False
    r = len(U[0]) if n*m > 0 else 0
    if len(V[0]) != r or len(W[0]) != r:
        return False
    if r == 0:
        return n*m == 0
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for _ in range(max_check):
        A = np.random.randint(-2, 3, (n, m))
        B = np.random.randint(-2, 3, (m, p))
        C_expected = A @ B
        
        # Compute C_calc using the decomposition
        # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        C_calc = np.zeros((n, p), dtype=np.float64)
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for r in range(r):
                        C_calc[i, k] += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
        
        if not np.allclose(C_calc, C_expected, atol=1e-9):
            return False
    return True

def _block_decompose_2x5x6():
    """
    Construct a block decomposition for 2x5x6 matrix multiplication.
    
    Strategy:
    - Split m=5 into m1=2, m2=3
    - Split p=6 into p1=2, p2=2, p3=2
    
    Block structure:
    C = [A1 A2] @ [B1; B2; B3]
       = A1*B1 + A2*B2 + A2*B3  (where A1 is 2x2, A2 is 2x3, B1 is 2x2, B2 is 3x2, B3 is 3x2)
    
    Wait, let me reconsider. We have:
    A is 2x5, B is 5x6
    Split A into [A_left (2x2), A_right (2x3)]
    Split B into [B_top (2x6); B_bottom (3x6)]
    
    Then C = A_left * B_top + A_right * B_bottom
    
    For A_left * B_top (2x2x6): split B_top into [B1 (2x2), B2 (2x2), B3 (2x2)]
    So A_left * B_top = A_left*B1 + A_left*B2 + A_left*B3
    Each A_left*Bk is 2x2x2, can use Strassen (rank 7)
    Total for this part: 3*7 = 21
    
    For A_right * B_bottom (2x3x6): 
    Split A_right into [AR1 (2x2), AR2 (2x1)]
    Split B_bottom into [BB1 (2x6); BB2 (1x6)]
    So A_right * B_bottom = AR1*BB1 + AR2*BB2
    AR1*BB1 is 2x2x6, split into three 2x2x2: 3*7 = 21
    AR2*BB2 is 2x1x6, schoolbook: 2*1*6 = 12
    Total for this part: 21 + 12 = 33
    
    Grand total: 21 + 33 = 54 < 60 (schoolbook)
    """
    n, m, p = 2, 5, 6
    
    # Get Strassen decomposition for 2x2x2
    U_str, V_str, W_str = _strassen_2x2x2()
    
    # We'll build the full decomposition by combining blocks
    total_rank = 0
    U_rows = []
    V_rows = []
    W_rows = []
    
    # Helper to add a block decomposition
    def add_block(U_block, V_block, W_block, i_offset, j_offset, k_offset):
        """
        Add a block decomposition to the global decomposition.
        
        U_block has shape (n_b * m_b, r_b) where n_b, m_b are block dimensions
        V_block has shape (m_b * p_b, r_b)
        W_block has shape (p_b * n_b, r_b)
        
        i_offset: row offset in A (0 or 1)
        j_offset: column offset in A / row offset in B
        k_offset: column offset in B / column offset in C
        """
        nonlocal total_rank, U_rows, V_rows, W_rows
        
        n_b = len(U_block) // (m_b if m_b > 0 else 1)
        # Actually, let's pass dimensions explicitly
        pass
    
    # Let me restructure. I'll build the full U, V, W directly.
    
    # Block 1: A_left (rows 0-1, cols 0-1) * B_top (rows 0-1, cols 0-5)
    # Split B_top into B1 (cols 0-1), B2 (cols 2-3), B3 (cols 4-5)
    
    # Sub-block 1.1: A_left * B1 (2x2x2)
    # A indices: i=0,1; j=0,1
    # B indices: j=0,1; k=0,1
    # C indices: i=0,1; k=0,1
    
    U_full = []
    V_full = []
    W_full = []
    rank_offset = 0
    
    def add_strassen_block(i_off, j_off, k_off):
        """Add a Strassen 2x2x2 block at given offsets."""
        nonlocal rank_offset
        for r in range(7):
            u_row = [0] * (total_rank + 7)
            v_row = [0] * (total_rank + 7)
            w_row = [0] * (total_rank + 7)
            
            # U: A entries at (i_off + i, j_off + j)
            for i in range(2):
                for j in range(2):
                    u_idx = (i_off + i) * 5 + (j_off + j)
                    if u_idx < len(U_full):
                        U_full[u_idx][rank_offset + r] = U_str[i*2+j][r]
            
            # V: B entries at (j_off + j, k_off + k)
            for j in range(2):
                for k in range(2):
                    v_idx = (j_off + j) * 6 + (k_off + k)
                    if v_idx < len(V_full):
                        V_full[v_idx][rank_offset + r] = V_str[j*2+k][r]
            
            # W: C entries at (i_off + i, k_off + k)
            for i in range(2):
                for k in range(2):
                    w_idx = (k_off + k) * 2 + (i_off + i)
                    if w_idx < len(W_full):
                        W_full[w_idx][rank_offset + r] = W_str[k*2+i][r]
        
        rank_offset += 7
    
    def add_schoolbook_block(n_b, m_b, p_b, i_off, j_off, k_off):
        """Add a schoolbook block at given offsets."""
        nonlocal rank_offset
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    u_idx = (i_off + i) * 5 + (j_off + j)
                    v_idx = (j_off + j) * 6 + (k_off + k)
                    w_idx = (k_off + k) * 2 + (i_off + i)
                    
                    if u_idx < len(U_full) and v_idx < len(V_full) and w_idx < len(W_full):
                        U_full[u_idx][rank_offset] = 1
                        V_full[v_idx][rank_offset] = 1
                        W_full[w_idx][rank_offset] = 1
                    rank_offset += 1
    
    # Initialize full matrices
    n_full, m_full, p_full = 2, 5, 6
    U_full = [[0]*54 for _ in range(n_full * m_full)]
    V_full = [[0]*54 for _ in range(m_full * p_full)]
    W_full = [[0]*54 for _ in range(p_full * n_full)]
    rank_offset = 0
    
    # Block 1: A_left (2x2) * B_top (2x6)
    # Split into three 2x2x2 blocks
    
    # 1.1: A_left * B1 (cols 0-1)
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U_full[i*5+j][rank_offset+r] = U_str[i*2+j][r]
        for j in range(2):
            for k in range(2):
                V_full[j*6+k][rank_offset+r] = V_str[j*2+k][r]
        for i in range(2):
            for k in range(2):
                W_full[k*2+i][rank_offset+r] = W_str[k*2+i][r]
    rank_offset += 7
    
    # 1.2: A_left * B2 (cols 2-3)
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U_full[i*5+j][rank_offset+r] = U_str[i*2+j][r]
        for j in range(2):
            for k in range(2):
                V_full[j*6+(k+2)][rank_offset+r] = V_str[j*2+k][r]
        for i in range(2):
            for k in range(2):
                W_full[(k+2)*2+i][rank_offset+r] = W_str[k*2+i][r]
    rank_offset += 7
    
    # 1.3: A_left * B3 (cols 4-5)
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U_full[i*5+j][rank_offset+r] = U_str[i*2+j][r]
        for j in range(2):
            for k in range(2):
                V_full[j*6+(k+4)][rank_offset+r] = V_str[j*2+k][r]
        for i in range(2):
            for k in range(2):
                W_full[(k+4)*2+i][rank_offset+r] = W_str[k*2+i][r]
    rank_offset += 7
    
    # Block 2: A_right (2x3) * B_bottom (3x6)
    # Split A_right into AR1 (2x2, cols 2-3) and AR2 (2x1, col 4)
    # Split B_bottom into BB1 (2x6, rows 2-3) and BB2 (1x6, row 4)
    
    # 2.1: AR1 (2x2) * BB1 (2x6)
    # Split BB1 into three 2x2 blocks
    
    # 2.1.1: AR1 * BB1_col0_1
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U_full[i*5+(j+2)][rank_offset+r] = U_str[i*2+j][r]
        for j in range(2):
            for k in range(2):
                V_full[(j+2)*6+k][rank_offset+r] = V_str[j*2+k][r]
        for i in range(2):
            for k in range(2):
                W_full[k*2+i][rank_offset+r] = W_str[k*2+i][r]
    rank_offset += 7
    
    # 2.1.2: AR1 * BB1_col2_3
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U_full[i*5+(j+2)][rank_offset+r] = U_str[i*2+j][r]
        for j in range(2):
            for k in range(2):
                V_full[(j+2)*6+(k+2)][rank_offset+r] = V_str[j*2+k][r]
        for i in range(2):
            for k in range(2):
                W_full[(k+2)*2+i][rank_offset+r] = W_str[k*2+i][r]
    rank_offset += 7
    
    # 2.1.3: AR1 * BB1_col4_5
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U_full[i*5+(j+2)][rank_offset+r] = U_str[i*2+j][r]
        for j in range(2):
            for k in range(2):
                V_full[(j+2)*6+(k+4)][rank_offset+r] = V_str[j*2+k][r]
        for i in range(2):
            for k in range(2):
                W_full[(k+4)*2+i][rank_offset+r] = W_str[k*2+i][r]
    rank_offset += 7
    
    # 2.2: AR2 (2x1) * BB2 (1x6) - schoolbook
    for i in range(2):
        for k in range(6):
            U_full[i*5+4][rank_offset] = 1
            V_full[4*6+k][rank_offset] = 1
            W_full[k*2+i][rank_offset] = 1
            rank_offset += 1
    
    # Verify rank_offset should be 54
    assert rank_offset == 54, f"Expected rank 54, got {rank_offset}"
    
    return U_full, V_full, W_full

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Initialize schoolbook decomposition
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": [row[:] for row in U], "V": [row[:] for row in V], "W": [row[:] for row in W]}
    best_rank = schoolbook_rank
    
    if time.time() > deadline - 5:
        return best_result
    
    # Try block decomposition with Strassen
    try:
        U_block, V_block, W_block = _block_decompose_2x5x6()
        
        # Verify the block decomposition
        if _verify(U_block, V_block, W_block, n=2, m=5, p=6, max_check=20):
            best_result = {"U": U_block, "V": V_block, "W": W_block}
            best_rank = len(U_block[0])
    except Exception as e:
        # If block decomposition fails, fall back to schoolbook
        pass
    
    if time.time() > deadline - 5:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
