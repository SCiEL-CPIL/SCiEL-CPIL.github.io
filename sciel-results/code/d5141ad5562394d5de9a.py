"""Matrix multiplication 4x4 * 4x6 tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_block_strassen():
    """
    Build block Strassen decomposition for 4x4 * 4x6.
    A is 2x2 blocks of 2x2, B is 2x2 blocks of 2x3.
    Each block product 2x2 * 2x3 uses inner Strassen:
      Split 2x2 * 2x3 into (2x2 * 2x2) + (2x2 * 2x1)
      2x2*2x2 uses Strassen rank 7
      2x2*2x1 uses schoolbook rank 4
      Total per block product: 11
    7 Strassen terms * 11 = 77 rank
    """
    n, m, p = 4, 4, 6
    rank = 77
    U = [[0] * rank for _ in range(n * m)]  # 16 x 77
    V = [[0] * rank for _ in range(m * p)]  # 24 x 77
    W = [[0] * rank for _ in range(p * n)]  # 24 x 77
    
    col = 0
    
    def add_scalar_mult(a_row, a_col, b_row, b_col, c_row, c_col, coeff=1):
        """Add a rank-1 term for scalar multiplication A[a_row,a_col]*B[b_row,b_col]->C[c_row,c_col]"""
        nonlocal col
        a_idx = a_row * m + a_col
        b_idx = b_row * p + b_col
        # W index: k*n + i where k is col in B (= c_col), i is row in A (= c_row)
        c_idx = c_col * n + c_row
        U[a_idx][col] = coeff
        V[b_idx][col] = 1
        W[c_idx][col] = 1
        col += 1
    
    def add_2x2_x_2x1(a_base_r, a_base_c, b_base_r, b_base_c, c_base_r, c_base_c, coeff=1):
        """
        Add schoolbook 2x2 * 2x1 = 4 scalar mults.
        A block at (a_base_r, a_base_c) is 2x2
        B block at (b_base_r, b_base_c) is 2x1
        C block at (c_base_r, c_base_c) is 2x1
        """
        nonlocal col
        for ar in range(2):
            for ac in range(2):
                for br in range(2):
                    if ac != br:
                        continue
                    a_row = a_base_r + ar
                    a_col = a_base_c + ac
                    b_row = b_base_r + br
                    b_col = b_base_c
                    c_row = c_base_r + ar
                    c_col = c_base_c
                    a_idx = a_row * m + a_col
                    b_idx = b_row * p + b_col
                    c_idx = c_col * n + c_row
                    U[a_idx][col] = coeff
                    V[b_idx][col] = 1
                    W[c_idx][col] = 1
                    col += 1
    
    def add_2x2_x_2x2_strassen(a_base_r, a_base_c, b_base_r, b_base_c, c_base_r, c_base_c, coeff=1):
        """
        Add Strassen 2x2 * 2x2 = rank 7 scalar mults.
        A block at (a_base_r, a_base_c) is 2x2
        B block at (b_base_r, b_base_c) is 2x2
        C block at (c_base_r, c_base_c) is 2x2
        """
        nonlocal col
        # Strassen formulas for 2x2 * 2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        # We need to expand each M into scalar mults
        # M1 = (A11+A22)(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
        # But in Strassen, we compute M1 as a single rank-1 term where
        # U has the sum pattern and V has the sum pattern
        
        # Actually for exact decomposition, each M is a product of two linear combinations.
        # M1: U combines A11 and A22, V combines B11 and B22
        # This gives 1 rank-1 term per M, but the U and V entries are sums.
        
        # Let me think more carefully. In tensor decomposition:
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        
        # For a 2x2*2x2 product, the tensor has entries T[a,b,c] = delta(a,b) for a,b,c in {0,1}
        # Strassen gives rank 7: 7 rank-1 tensors
        
        # M1 = (A11+A22)(B11+B22): contributes to C11, C22
        #   U: A11=1, A22=1; V: B11=1, B22=1; W: C11=1, C22=1
        # M2 = (A21+A22)B11: contributes to C21, C22
        #   U: A21=1, A22=1; V: B11=1; W: C21=1, C22=-1
        # M3 = A11(B12-B22): contributes to C12, C22
        #   U: A11=1; V: B12=1, B22=-1; W: C12=1, C22=1
        # M4 = A22(B21-B11): contributes to C11, C21
        #   U: A22=1; V: B21=1, B11=-1; W: C11=1, C21=1
        # M5 = (A11+A12)B22: contributes to C11, C12
        #   U: A11=1, A12=1; V: B22=1; W: C11=-1, C12=1
        # M6 = (A21-A11)(B11+B12): contributes to C22
        #   U: A21=1, A11=-1; V: B11=1, B12=1; W: C22=1
        # M7 = (A12-A22)(B21+B22): contributes to C11
        #   U: A12=1, A22=-1; V: B21=1, B22=1; W: C11=1
        
        # Local indices: A is 2x2 at (a_base_r, a_base_c), B is 2x2 at (b_base_r, b_base_c)
        # A11 = (a_base_r, a_base_c), A12 = (a_base_r, a_base_c+1)
        # A21 = (a_base_r+1, a_base_c), A22 = (a_base_r+1, a_base_c+1)
        # B11 = (b_base_r, b_base_c), B12 = (b_base_r, b_base_c+1)
        # B21 = (b_base_r+1, b_base_c), B22 = (b_base_r+1, b_base_c+1)
        # C11 = (c_base_r, c_base_c), C12 = (c_base_r, c_base_c+1)
        # C21 = (c_base_r+1, c_base_c), C22 = (c_base_r+1, c_base_c+1)
        
        def _add_rank1(u_entries, v_entries, w_entries):
            """u_entries: list of (a_row, a_col, coeff)
               v_entries: list of (b_row, b_col, coeff)
               w_entries: list of (c_row, c_col, coeff)
            """
            nonlocal col
            for (ar, ac, uc) in u_entries:
                if uc == 0: continue
                U[ar * m + ac][col] = coeff * uc
            for (br, bc, vc) in v_entries:
                if vc == 0: continue
                V[br * p + bc][col] = vc
            for (cr, cc, wc) in w_entries:
                if wc == 0: continue
                W[cc * n + cr][col] = wc
            col += 1
        
        A11 = (a_base_r, a_base_c)
        A12 = (a_base_r, a_base_c + 1)
        A21 = (a_base_r + 1, a_base_c)
        A22 = (a_base_r + 1, a_base_c + 1)
        B11 = (b_base_r, b_base_c)
        B12 = (b_base_r, b_base_c + 1)
        B21 = (b_base_r + 1, b_base_c)
        B22 = (b_base_r + 1, b_base_c + 1)
        C11 = (c_base_r, c_base_c)
        C12 = (c_base_r, c_base_c + 1)
        C21 = (c_base_r + 1, c_base_c)
        C22 = (c_base_r + 1, c_base_c + 1)
        
        # M1 = (A11+A22)(B11+B22) -> C11=1, C22=1
        _add_rank1(
            [(A11[0], A11[1], 1), (A22[0], A22[1], 1)],
            [(B11[0], B11[1], 1), (B22[0], B22[1], 1)],
            [(C11[0], C11[1], 1), (C22[0], C22[1], 1)]
        )
        # M2 = (A21+A22)B11 -> C21=1, C22=-1
        _add_rank1(
            [(A21[0], A21[1], 1), (A22[0], A22[1], 1)],
            [(B11[0], B11[1], 1)],
            [(C21[0], C21[1], 1), (C22[0], C22[1], -1)]
        )
        # M3 = A11(B12-B22) -> C12=1, C22=1
        _add_rank1(
            [(A11[0], A11[1], 1)],
            [(B12[0], B12[1], 1), (B22[0], B22[1], -1)],
            [(C12[0], C12[1], 1), (C22[0], C22[1], 1)]
        )
        # M4 = A22(B21-B11) -> C11=1, C21=1
        _add_rank1(
            [(A22[0], A22[1], 1)],
            [(B21[0], B21[1], 1), (B11[0], B11[1], -1)],
            [(C11[0], C11[1], 1), (C21[0], C21[1], 1)]
        )
        # M5 = (A11+A12)B22 -> C11=-1, C12=1
        _add_rank1(
            [(A11[0], A11[1], 1), (A12[0], A12[1], 1)],
            [(B22[0], B22[1], 1)],
            [(C11[0], C11[1], -1), (C12[0], C12[1], 1)]
        )
        # M6 = (A21-A11)(B11+B12) -> C22=1
        _add_rank1(
            [(A21[0], A21[1], 1), (A11[0], A11[1], -1)],
            [(B11[0], B11[1], 1), (B12[0], B12[1], 1)],
            [(C22[0], C22[1], 1)]
        )
        # M7 = (A12-A22)(B21+B22) -> C11=1
        _add_rank1(
            [(A12[0], A12[1], 1), (A22[0], A22[1], -1)],
            [(B21[0], B21[1], 1), (B22[0], B22[1], 1)],
            [(C11[0], C11[1], 1)]
        )
    
    def add_2x2_x_2x3(a_base_r, a_base_c, b_base_r, b_base_c, c_base_r, c_base_c, coeff=1):
        """
        Add 2x2 * 2x3 = (2x2*2x2) + (2x2*2x1) = 7 + 4 = 11 rank.
        A block at (a_base_r, a_base_c) is 2x2
        B block at (b_base_r, b_base_c) is 2x3
        C block at (c_base_r, c_base_c) is 2x3
        """
        # Split B into 2x2 (cols 0,1) and 2x1 (col 2)
        # 2x2 * 2x2 -> 2x2 (Strassen rank 7)
        add_2x2_x_2x2_strassen(a_base_r, a_base_c, b_base_r, b_base_c, c_base_r, c_base_c, coeff)
        # 2x2 * 2x1 -> 2x1 (schoolbook rank 4)
        add_2x2_x_2x1(a_base_r, a_base_c, b_base_r, b_base_c + 2, c_base_r, c_base_c + 2, coeff)
    
    # Block Strassen for 4x4 * 4x6:
    # A blocks: A11(0,0), A12(0,2), A21(2,0), A22(2,2) each 2x2
    # B blocks: B11(0,0), B12(0,3), B21(2,0), B22(2,3) each 2x3
    # C blocks: C11(0,0), C12(0,3), C21(2,0), C22(2,3) each 2x3
    
    # M1 = (A11+A22)(B11+B22)
    for (ar, ac, ac_coeff) in [(0, 0, 1), (2, 2, 1)]:
        for (br, bc, bc_coeff) in [(0, 0, 1), (2, 3, 1)]:
            add_2x2_x_2x3(ar, ac, br, bc, ar, bc, coeff=ac_coeff * bc_coeff)
    
    # M2 = (A21+A22)B11
    for (ar, ac, ac_coeff) in [(2, 0, 1), (2, 2, 1)]:
        add_2x2_x_2x3(ar, ac, 0, 0, ar, 0, coeff=ac_coeff)
    
    # M3 = A11(B12-B22)
    for (br, bc, bc_coeff) in [(0, 3, 1), (2, 3, -1)]:
        add_2x2_x_2x3(0, 0, br, bc, 0, bc, coeff=bc_coeff)
    
    # M4 = A22(B21-B11)
    for (br, bc, bc_coeff) in [(2, 0, 1), (0, 0, -1)]:
        add_2x2_x_2x3(2, 2, br, bc, 2, bc, coeff=bc_coeff)
    
    # M5 = (A11+A12)B22
    for (ar, ac, ac_coeff) in [(0, 0, 1), (0, 2, 1)]:
        add_2x2_x_2x3(ar, ac, 2, 3, ar, 3, coeff=ac_coeff)
    
    # M6 = (A21-A11)(B11+B12)
    for (ar, ac, ac_coeff) in [(2, 0, 1), (0, 0, -1)]:
        for (br, bc, bc_coeff) in [(0, 0, 1), (0, 3, 1)]:
            add_2x2_x_2x3(ar, ac, br, bc, ar, bc, coeff=ac_coeff * bc_coeff)
    
    # M7 = (A12-A22)(B21+B22)
    for (ar, ac, ac_coeff) in [(0, 2, 1), (2, 2, -1)]:
        for (br, bc, bc_coeff) in [(2, 0, 1), (2, 3, 1)]:
            add_2x2_x_2x3(ar, ac, br, bc, ar, bc, coeff=ac_coeff * bc_coeff)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(result):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = 4, 4, 6
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    
    # T[i,j,k] = delta(i,j) for A*B=C where C[i,k] = sum_j A[i,j]*B[j,k]
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    np.random.seed(42)
    for _ in range(100):
        i = np.random.randint(n)
        j = np.random.randint(m)
        k = np.random.randint(p)
        expected = 1.0 if i == j else 0.0
        actual = 0.0
        for c in range(U.shape[1]):
            actual += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        if abs(actual - expected) > 1e-9:
            return False, (i, j, k, expected, actual)
    return True, None

def _build_schoolbook():
    """Schoolbook decomposition: rank = n*m*p = 96"""
    n, m, p = 4, 4, 6
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # Try block Strassen (rank 77)
    result = _build_block_strassen()
    
    # Verify
    valid, info = _verify_decomposition(result)
    if not valid:
        # Fall back to schoolbook (rank 96)
        return _build_schoolbook()
    
    return result
# EVOLVE-BLOCK-END
