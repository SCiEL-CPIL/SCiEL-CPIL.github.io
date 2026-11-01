"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # We'll build U (10 x rank), V (30 x rank), W (12 x rank)
    # Start with empty, accumulate contributions
    
    U = {}  # sparse: (row, col) -> value
    V = {}
    W = {}
    
    rank = 0
    
    def add_U(row, col, val):
        key = (row, col)
        U[key] = U.get(key, 0) + val
    
    def add_V(row, col, val):
        key = (row, col)
        V[key] = V.get(key, 0) + val
    
    def add_W(row, col, val):
        key = (row, col)
        W[key] = W.get(key, 0) + val
    
    def strassen_2x2x2(a_col0, a_col1, b_row0, b_row1, k_col0, k_col1):
        """
        Compute contribution of A[:, a_col0:a_col1+1] * B[b_row0:b_row1+1, k_col0:k_col1+1]
        to C[:, k_col0:k_col1+1] using Strassen (rank 7).
        
        A is 2x2: rows 0,1; cols a_col0, a_col1
        B is 2x2: rows b_row0, b_row1; cols k_col0, k_col1
        C is 2x2: rows 0,1; cols k_col0, k_col1
        """
        global rank
        
        # Local A indices: (i, j) where i in {0,1}, j in {0,1}
        # Global A index: i*m + (a_col0 + j)
        # Local B indices: (j, k) where j in {0,1}, k in {0,1}
        # Global B index: (b_row0 + j)*p + (k_col0 + k)
        # Local C indices: (i, k) where i in {0,1}, k in {0,1}
        # Global C index: (k_col0 + k)*n + i
        
        a00 = 0*m + a_col0  # A[0, a_col0]
        a01 = 0*m + a_col1  # A[0, a_col1]
        a10 = 1*m + a_col0  # A[1, a_col0]
        a11 = 1*m + a_col1  # A[1, a_col1]
        
        b00 = b_row0*p + k_col0  # B[b_row0, k_col0]
        b01 = b_row0*p + k_col1  # B[b_row0, k_col1]
        b10 = b_row1*p + k_col0  # B[b_row1, k_col0]
        b11 = b_row1*p + k_col1  # B[b_row1, k_col1]
        
        c00 = k_col0*n + 0  # C[0, k_col0]
        c01 = k_col1*n + 0  # C[0, k_col1]
        c10 = k_col0*n + 1  # C[1, k_col0]
        c11 = k_col1*n + 1  # C[1, k_col1]
        
        # M1 = (A00+A10)*(B00+B10)
        r = rank; rank += 1
        add_U(a00, r, 1)
        add_U(a10, r, 1)
        add_V(b00, r, 1)
        add_V(b10, r, 1)
        add_W(c00, r, 1)   # C[0,0] += M1
        add_W(c11, r, 1)   # C[1,1] += M1
        
        # M2 = (A10+A11)*B00
        r = rank; rank += 1
        add_U(a10, r, 1)
        add_U(a11, r, 1)
        add_V(b00, r, 1)
        add_W(c10, r, 1)   # C[1,0] += M2
        
        # M3 = A00*(B01-B11)
        r = rank; rank += 1
        add_U(a00, r, 1)
        add_V(b01, r, 1)
        add_V(b11, r, -1)
        add_W(c01, r, 1)   # C[0,1] += M3
        
        # M4 = A11*(B10-B00)
        r = rank; rank += 1
        add_U(a11, r, 1)
        add_V(b10, r, 1)
        add_V(b00, r, -1)
        add_W(c00, r, 1)   # C[0,0] += M4
        add_W(c10, r, 1)   # C[1,0] += M4
        
        # M5 = (A00+A01)*B11
        r = rank; rank += 1
        add_U(a00, r, 1)
        add_U(a01, r, 1)
        add_V(b11, r, 1)
        add_W(c00, r, -1)  # C[0,0] -= M5
        add_W(c01, r, 1)   # C[0,1] += M5
        
        # M6 = (A10-A00)*(B00+B01)
        r = rank; rank += 1
        add_U(a10, r, 1)
        add_U(a00, r, -1)
        add_V(b00, r, 1)
        add_V(b01, r, 1)
        add_W(c11, r, 1)   # C[1,1] += M6
        
        # M7 = (A01-A11)*(B10+B11)
        r = rank; rank += 1
        add_U(a01, r, 1)
        add_U(a11, r, -1)
        add_V(b10, r, 1)
        add_V(b11, r, 1)
        add_W(c00, r, 1)   # C[0,0] += M7
    
    def schoolbook_2x3x2(a_col0, a_col1, a_col2, b_row0, b_row1, b_row2, k_col0, k_col1):
        """
        Compute contribution of A[:, a_col0:a_col2+1] * B[b_row0:b_row2+1, k_col0:k_col1+1]
        to C[:, k_col0:k_col1+1] using schoolbook (rank 12).
        
        A is 2x3: rows 0,1; cols a_col0, a_col1, a_col2
        B is 3x2: rows b_row0, b_row1, b_row2; cols k_col0, k_col1
        C is 2x2: rows 0,1; cols k_col0, k_col1
        """
        global rank
        
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    r = rank; rank += 1
                    a_idx = i*m + (a_col0 + j)
                    b_idx = (b_row0 + j)*p + (k_col0 + k)
                    c_idx = (k_col0 + k)*n + i
                    add_U(a_idx, r, 1)
                    add_V(b_idx, r, 1)
                    add_W(c_idx, r, 1)
    
    # Part 1: A1 (2x2, cols 0,1) * B1 (2x6, rows 0,1)
    # Split B1 columns into pairs: (0,1), (2,3), (4,5)
    for k0, k1 in [(0,1), (2,3), (4,5)]:
        strassen_2x2x2(0, 1, 0, 1, k0, k1)
    
    # Part 2: A2 (2x3, cols 2,3,4) * B2 (3x6, rows 2,3,4)
    # Split B2 columns into pairs: (0,1), (2,3), (4,5)
    for k0, k1 in [(0,1), (2,3), (4,5)]:
        schoolbook_2x3x2(2, 3, 4, 2, 3, 4, k0, k1)
    
    # Convert sparse to dense
    U_dense = [[0]*rank for _ in range(n*m)]
    V_dense = [[0]*rank for _ in range(m*p)]
    W_dense = [[0]*rank for _ in range(p*n)]
    
    for (row, col), val in U.items():
        U_dense[row][col] = val
    for (row, col), val in V.items():
        V_dense[row][col] = val
    for (row, col), val in W.items():
        W_dense[row][col] = val
    
    return {"U": U_dense, "V": V_dense, "W": W_dense}
# EVOLVE-BLOCK-END
