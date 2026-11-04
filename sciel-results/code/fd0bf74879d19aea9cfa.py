"""Structural decomposition of 4x4x6 matrix multiplication using Strassen 2x2 blocks."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 4x4 * 4x6 multiplication into three independent 4x4 * 4x2 subproblems.
    Each 4x4*4x2 is decomposed using two Strassen 2x2*2x2 products (rank 7 each).
    Total rank = 3 * 2 * 7 = 42.
    
    Wait, let me reconsider. 4x4 * 4x6:
    - Split A (4x4) into four 2x2 blocks: A = [[A11, A12], [A21, A22]]
    - Split B (4x6) into six 2x2 blocks: B = [[B11, B12, B13], [B21, B22, B23]]
    
    Actually, let's think more carefully.
    A is 4x4, B is 4x6.
    
    Split A into 2x2 blocks:
    A = [[A11, A12],
         [A21, A22]]
    where each Aij is 2x2.
    
    Split B into 2x2 blocks (4 rows, 6 cols):
    B = [[B11, B12, B13],
         [B21, B22, B23]]
    where each Bij is 2x2.
    
    C = A*B is 4x6:
    C = [[C11, C12, C13],
         [C21, C22, C23]]
    where each Cij is 2x2.
    
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C13 = A11*B13 + A12*B23
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    C23 = A21*B13 + A22*B23
    
    Each Cij is a sum of two 2x2 * 2x2 products.
    
    To compute each 2x2*2x2 product, we can use Strassen (rank 7).
    But we have 12 such products (4 A-blocks * 3 B-column-blocks... wait, no).
    
    Actually, we have 12 individual 2x2*2x2 matrix products to compute:
    A11*B11, A11*B12, A11*B13,
    A12*B21, A12*B22, A12*B23,
    A21*B11, A21*B12, A21*B13,
    A22*B21, A22*B22, A22*B23
    
    That's 12 products. If each uses Strassen rank 7, total rank = 12*7 = 84.
    
    But we can do better by sharing Strassen computations.
    
    Alternative approach: Split the 4x6 B into three 4x2 blocks.
    B = [B1, B2, B3] where each Bi is 4x2.
    
    Then C = [A*B1, A*B2, A*B3] where each A*Bi is 4x2.
    
    For each 4x4 * 4x2 product, we can use a decomposition.
    
    4x4 * 4x2:
    Split A into 2x2 blocks: A = [[A11, A12], [A21, A22]]
    Split B (4x2) into 2x2 blocks: B = [[B11], [B21]] where B11, B21 are 2x2.
    
    C = [[A11*B11 + A12*B21],
         [A21*B11 + A22*B21]]
    
    That's 4 individual 2x2*2x2 products per 4x4*4x2 subproblem.
    With Strassen, each 2x2*2x2 costs rank 7, so 4*7 = 28 per subproblem.
    Three subproblems: 3*28 = 84.
    
    This matches the knowledge from the run: rank 84.
    
    Now I need to construct the actual U, V, W matrices.
    
    Let me define the Strassen decomposition for 2x2*2x2:
    For X (2x2) and Y (2x2), Z = X*Y.
    
    Strassen:
    M1 = (X11+X22)*(Y11+Y22)
    M2 = (X21+X22)*Y11
    M3 = X11*(Y12-Y22)
    M4 = X22*(Y21-Y11)
    M5 = (X11+X12)*Y22
    M6 = (X21-X11)*(Y11+Y12)
    M7 = (X12-X22)*(Y21+Y22)
    
    Z11 = M1 + M4 - M5 + M7
    Z12 = M3 + M5
    Z21 = M2 + M4
    Z22 = M1 - M2 + M3 + M6
    
    Each Mi is a scalar (inner product of two vectors of length 2).
    
    For the full decomposition, I'll build U, V, W by composing these.
    
    Let me index carefully.
    
    U has shape (n*m, rank) = (16, rank)
    V has shape (m*p, rank) = (24, rank)
    W has shape (p*n, rank) = (24, rank)
    
    The tensor T[i,j,k] = delta[i,k] for matrix multiplication (A is n x m, B is m x p, C is n x p).
    Wait, the convention is: C[i,k] = sum_j A[i,j] * B[j,k]
    
    U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    So the decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Let me build this systematically.
    
    For each of the 3 subproblems (column blocks of B), and for each of the 4 2x2*2x2 products within each subproblem, and for each of the 7 Strassen steps, I create a rank-1 term.
    
    Subproblem s = 0, 1, 2 (corresponding to B's column blocks)
    Product p = 0, 1, 2, 3 (the four 2x2*2x2 products)
    Strassen step t = 0, 1, 2, 3, 4, 5, 6
    
    Total rank = 3 * 4 * 7 = 84.
    """
    import time
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strassen coefficients for 2x2*2x2
    # Each Strassen step t computes M_t = sum over (a,b) of coeff_a * X[a] * coeff_b * Y[b]
    # where X is a 2x2 matrix flattened, Y is a 2x2 matrix flattened.
    # X = [[X0, X1], [X2, X3]] (row-major: X0=X11, X1=X12, X2=X21, X3=X22)
    # Y = [[Y0, Y1], [Y2, Y3]]
    
    # M1 = (X0+X3)*(Y0+Y3)
    # M2 = (X2+X3)*Y0
    # M3 = X0*(Y1-Y3)
    # M4 = X3*(Y2-Y0)
    # M5 = (X0+X1)*Y3
    # M6 = (X2-X0)*(Y0+Y1)
    # M7 = (X1-X3)*(Y2+Y3)
    
    # For each Strassen step, we need:
    # - The coefficients for X (which rows of A to use)
    # - The coefficients for Y (which rows/cols of B to use)
    # - The contribution to Z (which elements of the 2x2 output)
    
    # Strassen step definitions:
    # t: (x_coeffs, y_coeffs, z_contrib)
    # x_coeffs: list of (x_index, coeff) for the 2x2 X matrix
    # y_coeffs: list of (y_index, coeff) for the 2x2 Y matrix
    # z_contrib: list of (z_index, coeff) for the 2x2 Z matrix
    
    strassen_steps = [
        # M1 = (X0+X3)*(Y0+Y3)
        {
            'x': [(0, 1), (3, 1)],
            'y': [(0, 1), (3, 1)],
            'z': [(0, 1), (3, 1)]  # Z11 += M1, Z22 += M1
        },
        # M2 = (X2+X3)*Y0
        {
            'x': [(2, 1), (3, 1)],
            'y': [(0, 1)],
            'z': [(2, 1), (3, -1)]  # Z21 += M2, Z22 -= M2
        },
        # M3 = X0*(Y1-Y3)
        {
            'x': [(0, 1)],
            'y': [(1, 1), (3, -1)],
            'z': [(1, 1), (3, 1)]  # Z12 += M3, Z22 += M3
        },
        # M4 = X3*(Y2-Y0)
        {
            'x': [(3, 1)],
            'y': [(2, 1), (0, -1)],
            'z': [(0, 1), (2, 1)]  # Z11 += M4, Z21 += M4
        },
        # M5 = (X0+X1)*Y3
        {
            'x': [(0, 1), (1, 1)],
            'y': [(3, 1)],
            'z': [(0, -1), (1, 1)]  # Z11 -= M5, Z12 += M5
        },
        # M6 = (X2-X0)*(Y0+Y1)
        {
            'x': [(2, 1), (0, -1)],
            'y': [(0, 1), (1, 1)],
            'z': [(3, 1)]  # Z22 += M6
        },
        # M7 = (X1-X3)*(Y2+Y3)
        {
            'x': [(1, 1), (3, -1)],
            'y': [(2, 1), (3, 1)],
            'z': [(0, 1)]  # Z11 += M7
        },
    ]
    
    # Now build the decomposition.
    # For each subproblem s (0,1,2), each product p (0,1,2,3), each Strassen step t (0..6):
    
    # Mapping from (product p) to (A_block, B_block):
    # p=0: A11 * B11 (A block row 0, B block row 0)
    # p=1: A12 * B21 (A block col 1, B block row 1) -- wait, let me re-examine
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C13 = A11*B13 + A12*B23
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # C23 = A21*B13 + A22*B23
    
    # For subproblem s (column block of B), the products are:
    # s=0: A11*B11, A12*B21, A21*B11, A22*B21 -> contributes to C11, C21
    # s=1: A11*B12, A12*B22, A21*B12, A22*B22 -> contributes to C12, C22
    # s=2: A11*B13, A12*B23, A21*B13, A22*B23 -> contributes to C13, C23
    
    # For subproblem s, the four products are indexed by (a_row, b_row) where:
    # a_row in {0, 1} (which 2x2 block row of A)
    # b_row in {0, 1} (which 2x2 block row of B)
    
    # Product index p:
    # p = a_row * 2 + b_row
    # p=0: a_row=0, b_row=0 -> A11 * B(s, row 0)
    # p=1: a_row=0, b_row=1 -> A12 * B(s, row 1)  -- wait, A12 is block (0,1), B row 1 is B2x
    # Actually: A block (a_row, a_col), B block (b_row, b_col)
    # For C[a_row, s] = sum over a_col of A[a_row, a_col] * B[s, a_col]... 
    
    # Let me be more explicit.
    # A is 4x4, split into 2x2 blocks:
    # A = [[A_00, A_01],
    #      [A_10, A_11]]
    # where A_ij is 2x2, with A_ij[r,c] = A[2*i+r, 2*j+c]
    
    # B is 4x6, split into 2x2 blocks:
    # B = [[B_00, B_01, B_02],
    #      [B_10, B_11, B_12]]
    # where B_ij is 2x2, with B_ij[r,c] = B[2*i+r, 2*j+c]
    
    # C = A*B, 4x6, split into 2x2 blocks:
    # C = [[C_00, C_01, C_02],
    #      [C_10, C_11, C_12]]
    # where C_ij[r,c] = C[2*i+r, 2*j+c]
    
    # C_ij = A_00 * B_ij + A_01 * B_{1,j}  -- NO
    # C_00 = A_00*B_00 + A_01*B_10
    # C_01 = A_00*B_01 + A_01*B_11
    # C_02 = A_00*B_02 + A_01*B_12
    # C_10 = A_10*B_00 + A_11*B_10
    # C_11 = A_10*B_01 + A_11*B_11
    # C_12 = A_10*B_02 + A_11*B_12
    
    # So for column block j of C (j=0,1,2), we have:
    # C_0j = A_00*B_0j + A_01*B_1j
    # C_1j = A_10*B_0j + A_11*B_1j
    
    # Four products per column block j:
    # (A_00, B_0j), (A_01, B_1j), (A_10, B_0j), (A_11, B_1j)
    
    # Let me define the mapping:
    # For subproblem j (0,1,2) and product index q (0,1,2,3):
    # q=0: A_block=(0,0), B_block=(0,j)
    # q=1: A_block=(0,1), B_block=(1,j)
    # q=2: A_block=(1,0), B_block=(0,j)
    # q=3: A_block=(1,1), B_block=(1,j)
    
    # A_block=(ar, ac) means A[2*ar:2*ar+2, 2*ac:2*ac+2]
    # B_block=(br, bc) means B[2*br:2*br+2, 2*bc:2*bc+2]
    
    # The output C_block for product (ar, ac) * (br, bc) contributes to C[ar, bc]
    # Specifically, A[ar,ac] * B[br,bc] contributes to C[ar,bc] if ac == br.
    # For q=0: ar=0, ac=0, br=0, bc=j -> ac==br (0==0), contributes to C[0,j]
    # For q=1: ar=0, ac=1, br=1, bc=j -> ac==br (1==1), contributes to C[0,j]
    # For q=2: ar=1, ac=0, br=0, bc=j -> ac==br (0==0), contributes to C[1,j]
    # For q=3: ar=1, ac=1, br=1, bc=j -> ac==br (1==1), contributes to C[1,j]
    
    # Good, this is consistent.
    
    # Now, for each (j, q, t), I create rank-1 terms.
    
    rank = 3 * 4 * 7
    U = [[0] * rank for _ in range(n * m)]  # 16 x rank
    V = [[0] * rank for _ in range(m * p)]  # 24 x rank
    W = [[0] * rank for _ in range(p * n)]  # 24 x rank
    
    col = 0
    for j in range(3):  # column block of B
        for q in range(4):  # product index
            # Determine A_block and B_block
            if q == 0:
                ar, ac = 0, 0
                br, bc = 0, j
            elif q == 1:
                ar, ac = 0, 1
                br, bc = 1, j
            elif q == 2:
                ar, ac = 1, 0
                br, bc = 0, j
            elif q == 3:
                ar, ac = 1, 1
                br, bc = 1, j
            
            # A_block is at rows [2*ar, 2*ar+2), cols [2*ac, 2*ac+2)
            # B_block is at rows [2*br, 2*br+2), cols [2*bc, 2*bc+2)
            
            for t in range(7):
                step = strassen_steps[t]
                
                # U: A_block indices
                # A_block[r, c] = A[2*ar+r, 2*ac+c]
                # In U, the row index is i*m + j where i is row of A, j is col of A.
                # So for A[2*ar+r, 2*ac+c], U row = (2*ar+r)*4 + (2*ac+c)
                
                for (x_idx, x_coeff) in step['x']:
                    # x_idx is 0,1,2,3 for the 2x2 X matrix
                    # x_idx=0: X[0,0], x_idx=1: X[0,1], x_idx=2: X[1,0], x_idx=3: X[1,1]
                    r = x_idx // 2
                    c = x_idx % 2
                    a_row = 2 * ar + r
                    a_col = 2 * ac + c
                    u_row = a_row * m + a_col
                    U[u_row][col] += x_coeff
                
                # V: B_block indices
                # B_block[r, c] = B[2*br+r, 2*bc+c]
                # In V, the row index is j*p + k where j is row of B, k is col of B.
                # So for B[2*br+r, 2*bc+c], V row = (2*br+r)*6 + (2*bc+c)
                
                for (y_idx, y_coeff) in step['y']:
                    r = y_idx // 2
                    c = y_idx % 2
                    b_row = 2 * br + r
                    b_col = 2 * bc + c
                    v_row = b_row * p + b_col
                    V[v_row][col] += y_coeff
                
                # W: C_block indices
                # C_block[r, c] = C[2*ar+r, 2*bc+c]  (output row from ar, output col from bc)
                # In W, the index is k*n + i where k is col of C, i is row of C.
                # So for C[2*ar+r, 2*bc+c], W index = (2*bc+c)*4 + (2*ar+r)
                
                for (z_idx, z_coeff) in step['z']:
                    r = z_idx // 2
                    c = z_idx % 2
                    c_row = 2 * ar + r
                    c_col = 2 * bc + c
                    w_idx = c_col * n + c_row
                    W[w_idx][col] += z_coeff
                
                col += 1
    
    if time.time() > deadline:
        # Should not happen for this fixed construction, but guard just in case
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
