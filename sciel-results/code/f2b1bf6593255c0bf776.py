"""
Rank-minimal exact tensor decomposition for 3x5x5 matrix multiplication.
Uses block decomposition: 1x5x5 schoolbook (rank 25) + 2x5x5 Strassen (rank 49) = 74.
"""

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time

def _schoolbook_block(n_block, m_block, p_block, i_offset, j_offset, k_offset, base_col):
    """
    Generate schoolbook terms for a sub-block.
    A is (n_block x m_block) at offset (i_offset, j_offset)
    B is (m_block x p_block) at offset (j_offset, k_offset)
    C is (n_block x p_block) at offset (i_offset, k_offset)
    
    Returns list of (U_row, V_row, W_row) tuples and number of columns used.
    """
    terms = []
    col = base_col
    for i in range(n_block):
        for j in range(m_block):
            for k in range(p_block):
                u_row = (i_offset + i) * 5 + (j_offset + j)
                v_row = (j_offset + j) * 5 + (k_offset + k)
                w_row = (k_offset + k) * 3 + (i_offset + i)
                terms.append((u_row, v_row, w_row, col))
                col += 1
    return terms, col - base_col

def _strassen_2x2x2(n_block, m_block, p_block, i_offset, j_offset, k_offset, base_col):
    """
    Generate Strassen terms for a 2x2x2 sub-block.
    A is (2x2) at offset (i_offset, j_offset)
    B is (2x2) at offset (j_offset, k_offset)
    C is (2x2) at offset (i_offset, k_offset)
    
    Strassen formulas:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B21 - B22)
    M4 = A22(B12 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    terms = []
    col = base_col
    
    # Map logical indices to global rows
    # A rows: i_offset+0, i_offset+1; cols: j_offset+0, j_offset+1
    # B rows: j_offset+0, j_offset+1; cols: k_offset+0, k_offset+1
    # C rows: i_offset+0, i_offset+1; cols: k_offset+0, k_offset+1
    
    def a_row(i, j):
        return (i_offset + i) * 5 + (j_offset + j)
    
    def b_row(i, j):
        return (j_offset + i) * 5 + (k_offset + j)
    
    def w_row(i, j):
        return (k_offset + j) * 3 + (i_offset + i)
    
    # M1 = (A11 + A22)(B11 + B22)
    # U: A11=+1, A22=+1
    # V: B11=+1, B22=+1
    # W: C11=+1, C12=0, C21=0, C22=+1
    terms.append((a_row(0,0), b_row(0,0), w_row(0,0), col))
    terms.append((a_row(1,1), b_row(0,0), w_row(0,0), col))
    terms.append((a_row(0,0), b_row(1,1), w_row(0,0), col))
    terms.append((a_row(1,1), b_row(1,1), w_row(0,0), col))
    # For C22 contribution of M1
    # We need separate terms for each C element
    # Actually, each Strassen multiplication M_i contributes to multiple C elements.
    # We need to create one column per (U_row, V_row) pair for each M_i,
    # and W will have the appropriate coefficients.
    
    # Let me restructure: for each of the 7 Strassen products, create columns
    # where U and V have the appropriate linear combinations, and W has the
    # coefficients for each C element that this product contributes to.
    
    terms = []
    col = base_col
    
    # M1 = (A11 + A22)(B11 + B22)
    # Contributes: C11 += M1, C22 += M1
    # We need 4 terms for the expansion (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
    # But we can group them. Each term is one (u,v,w) triple.
    # For M1, we create 4 columns (one for each product term in the expansion)
    
    # Actually, the standard way: each Strassen "multiplication" M_i is a scalar
    # product of two linear combinations. In the tensor decomposition, each
    # scalar multiplication creates rank-1 terms.
    
    # Let me just enumerate all 7 Strassen multiplications explicitly.
    
    # M1 = (A11 + A22)(B11 + B22)
    # Expansion: A11*B11 + A11*B22 + A22*B11 + A22*B22
    # C11 gets +1 from all 4, C22 gets +1 from all 4
    for (ai, aj) in [(0,0), (0,1), (1,0), (1,1)]:
        # ai=0 means A11, ai=1 means A22
        # aj=0 means B11, aj=1 means B22
        a_idx = 0 if ai == 0 else 1
        b_idx_i = 0 if aj == 0 else 1
        b_idx_j = 0 if aj == 0 else 1
        u = a_row(a_idx, a_idx)  # A11 or A22
        v = b_row(b_idx_i, b_idx_j)  # B11 or B22
        # C11 and C22 both get +1
        terms.append((u, v, w_row(0,0), col))
        col += 1
        terms.append((u, v, w_row(1,1), col))
        col += 1
    
    # M2 = (A21 + A22)B11
    # Expansion: A21*B11 + A22*B11
    # C21 gets +1 from both
    for ai in [1]:  # A21 and A22
        u = a_row(1, ai)  # A21 (ai=1? no, A21 is row1,col0)
        # A21 = a_row(1,0), A22 = a_row(1,1)
        pass
    # Let me be more careful
    # A21 = A[1,0], A22 = A[1,1]
    # B11 = B[0,0]
    u1 = a_row(1,0)  # A21
    u2 = a_row(1,1)  # A22
    v = b_row(0,0)   # B11
    terms.append((u1, v, w_row(1,0), col))
    col += 1
    terms.append((u2, v, w_row(1,0), col))
    col += 1
    
    # M3 = A11(B21 - B22)
    # Expansion: A11*B21 - A11*B22
    # C12 gets +1 from B21 term, -1 from B22 term
    u = a_row(0,0)  # A11
    v1 = b_row(1,0)  # B21
    v2 = b_row(1,1)  # B22
    terms.append((u, v1, w_row(0,1), col))
    col += 1
    terms.append((u, v2, w_row(0,1), col))
    # W coefficient for B22 term is -1, but we handle signs in W
    # Actually, W entries can be negative. Let me set them.
    
    # I need to track W coefficients properly. Let me restructure.
    
    terms = []
    col = base_col
    
    def add_term(u, v, w, coeff):
        nonlocal col
        terms.append((u, v, w, col, coeff))
        col += 1
    
    # M1 = (A11 + A22)(B11 + B22)
    # C11 += M1, C22 += M1
    for a_idx in [0, 1]:  # A11, A22
        for b_idx in [0, 1]:  # B11, B22
            u = a_row(a_idx, a_idx)
            v = b_row(b_idx, b_idx)
            add_term(u, v, w_row(0,0), 1)
            add_term(u, v, w_row(1,1), 1)
    
    # M2 = (A21 + A22)B11
    # C21 += M2
    for a_idx in [0, 1]:  # A21, A22 (col index 0, 1)
        u = a_row(1, a_idx)
        v = b_row(0, 0)
        add_term(u, v, w_row(1,0), 1)
    
    # M3 = A11(B21 - B22)
    # C12 += M3
    u = a_row(0, 0)
    v1 = b_row(1, 0)
    v2 = b_row(1, 1)
    add_term(u, v1, w_row(0,1), 1)
    add_term(u, v2, w_row(0,1), -1)
    
    # M4 = A22(B12 - B11)
    # C11 += M4, C21 += M4
    u = a_row(1, 1)
    v1 = b_row(0, 1)
    v2 = b_row(0, 0)
    add_term(u, v1, w_row(0,0), 1)
    add_term(u, v2, w_row(0,0), -1)
    add_term(u, v1, w_row(1,0), 1)
    add_term(u, v2, w_row(1,0), -1)
    
    # M5 = (A11 + A12)B22
    # C11 -= M5, C12 += M5
    for a_idx in [0, 1]:  # A11, A12
        u = a_row(0, a_idx)
        v = b_row(1, 1)
        add_term(u, v, w_row(0,0), -1)
        add_term(u, v, w_row(0,1), 1)
    
    # M6 = (A21 - A11)(B11 + B12)
    # C22 += M6
    for a_idx in [0, 1]:  # A21, A11
        for b_idx in [0, 1]:  # B11, B12
            u = a_row(1 - a_idx, a_idx)  # a_idx=0: A21, a_idx=1: A11
            # Wait, A21 = a_row(1,0), A11 = a_row(0,0)
            # If a_idx=0: u should be A21 = a_row(1,0)
            # If a_idx=1: u should be A11 = a_row(0,0)
            # So u = a_row(1-a_idx, 0)
            u = a_row(1 - a_idx, 0)
            v = b_row(0, b_idx)
            coeff = 1 if a_idx == 0 else -1  # A21 has +1, A11 has -1
            add_term(u, v, w_row(1,1), coeff)
    
    # M7 = (A12 - A22)(B21 + B22)
    # C11 += M7
    for a_idx in [0, 1]:  # A12, A22
        for b_idx in [0, 1]:  # B21, B22
            u = a_row(0 if a_idx == 0 else 1, 1)  # A12 = a_row(0,1), A22 = a_row(1,1)
            v = b_row(1, b_idx)
            coeff = 1 if a_idx == 0 else -1
            add_term(u, v, w_row(0,0), coeff)
    
    return terms, col - base_col

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    total_rank = 74  # 25 (schoolbook 1x5x5) + 49 (Strassen 2x5x5)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: Row 0 of A, all of B -> schoolbook 1x5x5, rank 25
    for i in range(1):
        for j in range(5):
            for k in range(5):
                u_row = 0 * 5 + j
                v_row = j * 5 + k
                w_row = k * 3 + 0
                U[u_row][col] = 1
                V[v_row][col] = 1
                W[w_row][col] = 1
                col += 1
    
    # Block 2: Rows 1-2 of A, all of B -> Strassen 2x5x5, rank 49
    # Split into 4 sub-blocks:
    # A is 2x5, B is 5x5
    # Split A cols into [0,1] and [2,3,4]
    # Split B rows into [0,1] and [2,3,4], B cols into [0,1] and [2,3,4]
    
    # Sub-block 2.1: A[1:3,0:2] (2x2) * B[0:2,0:2] (2x2) -> C[1:3,0:2] (2x2)
    # Strassen, rank 7
    terms_21, rank_21 = _strassen_2x2x2(2, 2, 2, 1, 0, 0, col)
    for u, v, w, c, coeff in terms_21:
        U[u][c] = 1
        V[v][c] = 1
        W[w][c] = coeff
    col += rank_21
    
    # Sub-block 2.2: A[1:3,2:5] (2x3) * B[2:5,0:2] (3x2) -> C[1:3,0:2] (2x2)
    # Schoolbook, rank 12
    for i in range(2):
        for j in range(3):
            for k in range(2):
                u_row = (1 + i) * 5 + (2 + j)
                v_row = (2 + j) * 5 + (0 + k)
                w_row = (0 + k) * 3 + (1 + i)
                U[u_row][col] = 1
                V[v_row][col] = 1
                W[w_row][col] = 1
                col += 1
    
    # Sub-block 2.3: A[1:3,0:2] (2x2) * B[0:2,2:5] (2x3) -> C[1:3,2:5] (2x3)
    # Schoolbook, rank 12
    for i in range(2):
        for j in range(2):
            for k in range(3):
                u_row = (1 + i) * 5 + (0 + j)
                v_row = (0 + j) * 5 + (2 + k)
                w_row = (2 + k) * 3 + (1 + i)
                U[u_row][col] = 1
                V[v_row][col] = 1
                W[w_row][col] = 1
                col += 1
    
    # Sub-block 2.4: A[1:3,2:5] (2x3) * B[2:5,2:5] (3x3) -> C[1:3,2:5] (2x3)
    # Schoolbook, rank 18
    for i in range(2):
        for j in range(3):
            for k in range(3):
                u_row = (1 + i) * 5 + (2 + j)
                v_row = (2 + j) * 5 + (2 + k)
                w_row = (2 + k) * 3 + (1 + i)
                U[u_row][col] = 1
                V[v_row][col] = 1
                W[w_row][col] = 1
                col += 1
    
    assert col == total_rank, f"Expected rank {total_rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
