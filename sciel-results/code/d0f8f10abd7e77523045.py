"""Rewritten program using block decomposition with Strassen for 2x3x5 matrix multiplication."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    # Total rank estimate:
    # Block 1: 2x2x2 (cols 0-1 of B, rows 0-1 of B) -> Rank 7
    # Block 2: 2x2x2 (cols 2-3 of B, rows 0-1 of B) -> Rank 7
    # Block 3: 2x2x1 (col 4 of B, rows 0-1 of B)   -> Rank 4
    # Block 4: 2x1x5 (col 2 of A, all cols of B)     -> Rank 10
    # Total = 7 + 7 + 4 + 10 = 28
    
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # shape (6, 28)
    V = [[0] * rank for _ in range(m * p)]  # shape (15, 28)
    W = [[0] * rank for _ in range(p * n)]  # shape (10, 28)
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen_222(i_offset, j_offset, k_offset):
        nonlocal col
        # i_offset: start row of A (0-1)
        # j_offset: start col of A (0-1)
        # k_offset: start col of B (0-1)
        
        # A indices: (i_offset, j_offset), (i_offset, j_offset+1), (i_offset+1, j_offset), (i_offset+1, j_offset+1)
        # B indices: (j_offset, k_offset), (j_offset, k_offset+1), (j_offset+1, k_offset), (j_offset+1, k_offset+1)
        # C indices: (i_offset, k_offset), (i_offset, k_offset+1), (i_offset+1, k_offset), (i_offset+1, k_offset+1)
        
        # M0 = (A00 + A11)(B00 + B11)
        # M1 = (A10 + A11)B00
        # M2 = A00(B01 - B11)
        # M3 = A11(B10 - B00)
        # M4 = (A00 + A01)B11
        # M5 = (A10 - A00)(B00 + B01)
        # M6 = (A01 - A11)(B10 + B11)
        
        # C00 = M0 + M3 - M4 + M6
        # C01 = M2 + M4
        # C10 = M1 + M3
        # C11 = M0 - M1 + M2 + M5
        
        # Define the 7 multiplications
        # Each mult is (u_coeffs, v_coeffs, w_coeffs)
        # u_coeffs: dict { (i,j): val }
        # v_coeffs: dict { (j,k): val }
        # w_coeffs: dict { (i,k): val }
        
        muls = [
            # M0
            ({(i_offset, j_offset): 1, (i_offset+1, j_offset+1): 1}, 
             {(j_offset, k_offset): 1, (j_offset+1, k_offset+1): 1}, 
             {(i_offset, k_offset): 1, (i_offset+1, k_offset+1): 1}),
            # M1
            ({(i_offset+1, j_offset): 1, (i_offset+1, j_offset+1): 1}, 
             {(j_offset, k_offset): 1}, 
             {(i_offset+1, k_offset): 1}),
            # M2
            ({(i_offset, j_offset): 1}, 
             {(j_offset, k_offset+1): 1, (j_offset+1, k_offset+1): -1}, 
             {(i_offset, k_offset+1): 1}),
            # M3
            ({(i_offset+1, j_offset+1): 1}, 
             {(j_offset+1, k_offset): 1, (j_offset, k_offset): -1}, 
             {(i_offset, k_offset): 1, (i_offset+1, k_offset): 1}),
            # M4
            ({(i_offset, j_offset): 1, (i_offset, j_offset+1): 1}, 
             {(j_offset+1, k_offset+1): 1}, 
             {(i_offset, k_offset): -1, (i_offset, k_offset+1): 1}),
            # M5
            ({(i_offset+1, j_offset): 1, (i_offset, j_offset): -1}, 
             {(j_offset, k_offset): 1, (j_offset, k_offset+1): 1}, 
             {(i_offset+1, k_offset+1): 1}),
            # M6
            ({(i_offset, j_offset+1): 1, (i_offset+1, j_offset+1): -1}, 
             {(j_offset+1, k_offset): 1, (j_offset+1, k_offset+1): 1}, 
             {(i_offset, k_offset): 1, (i_offset+1, k_offset+1): 1}),
        ]
        
        for u_c, v_c, w_c in muls:
            for (i, j), val in u_c.items():
                U[i*m + j][col] += val
            for (j, k), val in v_c.items():
                V[j*p + k][col] += val
            for (i, k), val in w_c.items():
                W[k*n + i][col] += val
            col += 1
            
    # Helper to add a 2x2x1 schoolbook block
    def add_schoolbook_221(i_offset, j_offset, k_offset):
        nonlocal col
        # A: 2x2, B: 2x1, C: 2x1
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        # 4 multiplications
        muls = [
            ({(i_offset, j_offset): 1}, {(j_offset, k_offset): 1}, {(i_offset, k_offset): 1}),
            ({(i_offset, j_offset+1): 1}, {(j_offset+1, k_offset): 1}, {(i_offset, k_offset): 1}),
            ({(i_offset+1, j_offset): 1}, {(j_offset, k_offset): 1}, {(i_offset+1, k_offset): 1}),
            ({(i_offset+1, j_offset+1): 1}, {(j_offset+1, k_offset): 1}, {(i_offset+1, k_offset): 1}),
        ]
        for u_c, v_c, w_c in muls:
            for (i, j), val in u_c.items():
                U[i*m + j][col] += val
            for (j, k), val in v_c.items():
                V[j*p + k][col] += val
            for (i, k), val in w_c.items():
                W[k*n + i][col] += val
            col += 1

    # Helper to add a 2x1x5 schoolbook block
    def add_schoolbook_215(i_offset, j_offset, k_offset):
        nonlocal col
        # A: 2x1, B: 1x5, C: 2x5
        # C_i,k = A_i,j B_j,k
        # 10 multiplications
        for i in range(2):
            for k in range(5):
                U[(i_offset+i)*m + (j_offset)][col] = 1
                V[(j_offset)*p + (k_offset+k)][col] = 1
                W[(k_offset+k)*n + (i_offset+i)][col] = 1
                col += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # i=0, j=0, k=0
    add_strassen_222(0, 0, 0)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # i=0, j=0, k=2
    add_strassen_222(0, 0, 2)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5]
    # i=0, j=0, k=4
    add_schoolbook_221(0, 0, 4)
    
    # Block 4: A[0:2, 2:3] * B[2:3, 0:5] -> C[0:2, 0:5]
    # i=0, j=2, k=0
    add_schoolbook_215(0, 2, 0)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
